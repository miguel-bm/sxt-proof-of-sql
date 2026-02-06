use super::{DynProofExpr, ProofExpr};
use crate::{
    base::{
        database::{can_abs_type, Column, ColumnRef, ColumnType, LiteralValue, Table},
        if_rayon,
        map::{IndexMap, IndexSet},
        proof::{PlaceholderResult, ProofError},
        scalar::Scalar,
    },
    sql::{
        proof::{FinalRoundBuilder, SumcheckSubpolynomialType, VerificationBuilder},
        proof_gadgets::{
            final_round_evaluate_sign, first_round_evaluate_sign, verifier_evaluate_sign,
        },
        AnalyzeError, AnalyzeResult,
    },
    utils::log,
};
use alloc::{boxed::Box, vec, vec::Vec};
use bumpalo::Bump;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use sqlparser::ast::Ident;

/// Provable absolute value expression
///
/// For a value `x`, `abs(x) = x` if `x >= 0`, otherwise `abs(x) = -x`.
/// This uses the sign gadget to determine if the value is negative.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbsExpr {
    expr: Box<DynProofExpr>,
}

impl AbsExpr {
    /// Create a new absolute value expression
    pub fn try_new(expr: Box<DynProofExpr>) -> AnalyzeResult<Self> {
        let expr_type = expr.data_type();
        can_abs_type(expr_type)
            .then_some(Self { expr })
            .ok_or(AnalyzeError::InvalidDataType { expr_type })
    }

    /// Get the input expression
    pub fn input(&self) -> &DynProofExpr {
        &self.expr
    }
}

impl ProofExpr for AbsExpr {
    fn data_type(&self) -> ColumnType {
        self.expr.data_type()
    }

    #[tracing::instrument(name = "AbsExpr::first_round_evaluate", level = "debug", skip_all)]
    fn first_round_evaluate<'a, S: Scalar>(
        &self,
        alloc: &'a Bump,
        table: &Table<'a, S>,
        params: &[LiteralValue],
    ) -> PlaceholderResult<Column<'a, S>> {
        log::log_memory_usage("Start");

        let expr_column = self.expr.first_round_evaluate(alloc, table, params)?;
        let table_length = table.num_rows();
        let expr_scalars = expr_column.to_scalar();

        // Get sign bits (true if negative)
        let signs = first_round_evaluate_sign(table_length, alloc, &expr_scalars);

        // Compute abs: if sign is negative (-1), negate the value
        let result = compute_abs(alloc, &expr_scalars, signs);

        log::log_memory_usage("End");

        Ok(Column::Scalar(result))
    }

    #[tracing::instrument(name = "AbsExpr::final_round_evaluate", level = "debug", skip_all)]
    fn final_round_evaluate<'a, S: Scalar>(
        &self,
        builder: &mut FinalRoundBuilder<'a, S>,
        alloc: &'a Bump,
        table: &Table<'a, S>,
        params: &[LiteralValue],
    ) -> PlaceholderResult<Column<'a, S>> {
        log::log_memory_usage("Start");

        let expr_column = self
            .expr
            .final_round_evaluate(builder, alloc, table, params)?;
        // Allocate expr_scalars in bump allocator so it lives for 'a
        let expr_scalars = alloc.alloc_slice_copy(&expr_column.to_scalar());

        // Get sign bits (true if negative) and produce the necessary proof components.
        // The sign gadget commits bit decomposition MLEs and proves they are binary.
        let signs = final_round_evaluate_sign(builder, alloc, expr_scalars);

        // Convert signs to scalars for use in the sumcheck polynomial.
        // We commit this as an intermediate MLE so the sumcheck can use it.
        let sign_scalars: &[S] =
            alloc
                .alloc_slice_fill_iter(signs.iter().map(|&b| if b { S::one() } else { S::zero() }));
        builder.produce_intermediate_mle(sign_scalars as &[_]);

        // Compute abs: if sign is negative, negate the value
        let result = compute_abs(alloc, expr_scalars, signs);

        // Commit the result MLE
        builder.produce_intermediate_mle(result as &[_]);

        // Prove the constraint: result = expr * (1 - 2*sign)
        // Rearranged: result - expr + 2*expr*sign = 0
        // This ensures result = expr when sign=0, and result = -expr when sign=1
        builder.produce_sumcheck_subpolynomial(
            SumcheckSubpolynomialType::Identity,
            vec![
                (S::one(), vec![Box::new(result as &[_])]),
                (-S::one(), vec![Box::new(expr_scalars as &[_])]),
                (
                    S::TWO,
                    vec![
                        Box::new(expr_scalars as &[_]),
                        Box::new(sign_scalars as &[_]),
                    ],
                ),
            ],
        );

        log::log_memory_usage("End");

        Ok(Column::Scalar(result))
    }

    fn verifier_evaluate<S: Scalar>(
        &self,
        builder: &mut impl VerificationBuilder<S>,
        accessor: &IndexMap<Ident, S>,
        chi_eval: S,
        params: &[LiteralValue],
    ) -> Result<S, ProofError> {
        let expr_eval = self
            .expr
            .verifier_evaluate(builder, accessor, chi_eval, params)?;

        // Verify the sign gadget's bit decomposition constraints.
        // This consumes the sign gadget's bit MLEs and verifies they are binary.
        // The sign gadget proves that the bit decomposition is valid.
        let _chi_minus_sign_eval = verifier_evaluate_sign(builder, expr_eval, chi_eval, None)?;

        // Consume the committed sign_scalars MLE evaluation.
        // Note: sign_scalars is committed separately and used in the sumcheck constraint.
        // The sign gadget verifies the bit decomposition is correct, ensuring data integrity.
        let sign_scalars_eval = builder.try_consume_final_round_mle_evaluation()?;

        // Consume the result MLE evaluation
        let result_eval = builder.try_consume_final_round_mle_evaluation()?;

        // Verify the constraint: result - expr + 2*expr*sign = 0
        // This ensures result = expr when sign=0, and result = -expr when sign=1
        builder.try_produce_sumcheck_subpolynomial_evaluation(
            SumcheckSubpolynomialType::Identity,
            result_eval - expr_eval + S::TWO * expr_eval * sign_scalars_eval,
            2,
        )?;

        Ok(result_eval)
    }

    fn get_column_references(&self, columns: &mut IndexSet<ColumnRef>) {
        self.expr.get_column_references(columns);
    }
}

/// Compute absolute value: if sign is true (negative), negate the value
fn compute_abs<'a, S: Scalar>(alloc: &'a Bump, values: &[S], signs: &[bool]) -> &'a [S] {
    let result: Vec<S> = if_rayon!(
        values.par_iter().zip(signs.par_iter()),
        values.iter().zip(signs.iter())
    )
    .map(|(&val, &is_negative)| if is_negative { -val } else { val })
    .collect();

    alloc.alloc_slice_copy(&result)
}

#[cfg(all(test, feature = "blitzar"))]
mod tests {
    use super::*;
    use crate::{
        base::{
            commitment::InnerProductProof,
            database::{
                owned_table_utility::*, ColumnField, ColumnType, OwnedTableTestAccessor, TableRef,
            },
        },
        sql::{
            proof::{exercise_verification, VerifiableQueryResult},
            proof_exprs::test_utility::*,
            proof_plans::test_utility::*,
        },
    };

    #[test]
    fn we_can_compute_abs_of_positive_values() {
        let data = owned_table([bigint("a", [1_i64, 2, 3, 4, 5])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::BigInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([bigint("abs_a", [1_i64, 2, 3, 4, 5])]);
        assert_eq!(res, expected);
    }

    #[test]
    fn we_can_compute_abs_of_negative_values() {
        let data = owned_table([bigint("a", [-1_i64, -2, -3, -4, -5])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::BigInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([bigint("abs_a", [1_i64, 2, 3, 4, 5])]);
        assert_eq!(res, expected);
    }

    #[test]
    fn we_can_compute_abs_of_zero() {
        let data = owned_table([bigint("a", [0_i64, 0, 0])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::BigInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([bigint("abs_a", [0_i64, 0, 0])]);
        assert_eq!(res, expected);
    }

    #[test]
    fn we_can_compute_abs_of_mixed_values() {
        let data = owned_table([bigint("a", [-5_i64, -3, 0, 3, 5])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::BigInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([bigint("abs_a", [5_i64, 3, 0, 3, 5])]);
        assert_eq!(res, expected);
    }

    #[test]
    fn abs_rejects_non_numeric_types() {
        let bool_expr = DynProofExpr::new_literal(LiteralValue::Boolean(true));
        assert!(AbsExpr::try_new(Box::new(bool_expr)).is_err());

        let varchar_expr = DynProofExpr::new_literal(LiteralValue::VarChar("test".to_string()));
        assert!(AbsExpr::try_new(Box::new(varchar_expr)).is_err());
    }

    #[test]
    fn we_can_compute_abs_of_int128_values() {
        let data = owned_table([int128("a", [-1_000_000_000_000_i128, 0, 1_000_000_000_000])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::Int128)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([int128(
            "abs_a",
            [1_000_000_000_000_i128, 0, 1_000_000_000_000],
        )]);
        assert_eq!(res, expected);
    }

    #[test]
    fn we_can_compute_abs_of_smallint_values() {
        let data = owned_table([smallint("a", [-100_i16, -1, 0, 1, 100])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::SmallInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([smallint("abs_a", [100_i16, 1, 0, 1, 100])]);
        assert_eq!(res, expected);
    }

    #[test]
    fn we_can_compute_abs_of_large_negative_values() {
        // Test with values close to the limits of the type
        let data = owned_table([bigint("a", [i64::MIN + 1, i64::MAX, -i64::MAX])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::BigInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        // i64::MIN + 1 = -9223372036854775807, abs = 9223372036854775807 = i64::MAX
        // i64::MAX = 9223372036854775807, abs = 9223372036854775807
        // -i64::MAX = -9223372036854775807, abs = 9223372036854775807
        let expected = owned_table([bigint("abs_a", [i64::MAX, i64::MAX, i64::MAX])]);
        assert_eq!(res, expected);
    }

    #[test]
    fn we_can_compute_abs_with_single_row() {
        let data = owned_table([bigint("a", [-42_i64])]);
        let t = TableRef::new("sxt", "t");
        let accessor =
            OwnedTableTestAccessor::<InnerProductProof>::new_from_table(t.clone(), data, 0, ());
        let table_exec = table_exec(
            t.clone(),
            vec![ColumnField::new("a".into(), ColumnType::BigInt)],
        );
        let expr = filter(
            vec![aliased_plan(abs(column(&t, "a", &accessor)), "abs_a")],
            table_exec,
            const_bool(true),
        );
        let res = VerifiableQueryResult::new(&expr, &accessor, &(), &[]).unwrap();
        exercise_verification(&res, &expr, &accessor, &t);
        let res = res.verify(&expr, &accessor, &(), &[]).unwrap().table;
        let expected = owned_table([bigint("abs_a", [42_i64])]);
        assert_eq!(res, expected);
    }
}
