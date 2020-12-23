# Generate constraint data for percentages
PERCENTAGE=0.05
SEED=42
function generate_for_percentage() {
    PERCENTAGEREP=${PERCENTAGE/\./_}
    mkdir "oov_test_$PERCENTAGEREP/"
    CONSTRAINTS=synonyms.txt
    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    CONSTRAINTS=antonyms.txt
    python oov_cutter_slsv.py --target_file testing/SimLex-999.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv.py --target_file testing/SimVerb-3500.txt --percentage_to_leave $PERCENTAGE --seed $SEED --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimLex-999_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
    python oov_cutter_slsv_constraints.py --seen_words "oov_test_$PERCENTAGEREP/SimVerb-3500_cut_to_$PERCENTAGEREP.txt" --all_constraints $CONSTRAINTS --output_dir "oov_test_$PERCENTAGEREP/"
}
PERCENTAGE=0.05
generate_for_percentage
PERCENTAGE=0.1
generate_for_percentage
PERCENTAGE=0.25
generate_for_percentage
PERCENTAGE=0.5
generate_for_percentage
PERCENTAGE=0.75
generate_for_percentage
PERCENTAGE=1.0
generate_for_percentage