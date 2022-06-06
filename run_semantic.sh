#!bin/bash
# conda deactivate
# conda activate zerospeech2021
DATASET=/work/vjsalt22/dataset/zerospeech2021
# for idx in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12"
#do 
SUBMISSION=/work/vjsalt22/poheng/zerospeech2021
echo $SUBMISSION
# zerospeech2021-validate --only-dev --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION
# echo "Validation done!"
zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION
mv score_semantic_dev_correlation.csv $SUBMISSION
mv score_semantic_dev_pairs.csv $SUBMISSION
# done
echo "Evaluation done!"
