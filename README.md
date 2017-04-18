repository for the files used in trec2015 experiments

files
-----
- utils/runner.py: main *runner* file
- utils/sample_eval.pl: evaluation file for inferred measures (infAP, infNDCG)
- utils/cnnclf/data_prep.py: functions for data preprocessing
- utils/cnnclf/inspect_checkpoint.py: data utility file from TensorFlow Python
  script
- utils/cnnclf/model.py: MeshCNN model definition
- utils/cnnclf/trec_classify.py: runs inference in MeshCNN classifier
- utils/share/stopword-list.txt: custom stopwords list file for TREC
- utils/share/terrier.properties: config file for Terrier system settings
