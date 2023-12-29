# Identify_Speaker_Voice

  python3.8 -m venv venv && source venv/bin/activate && pip install -U pip setuptools wheel

### Windows

  pip install -r requirements.txt

## Linux or Mac 

  pip3 install -r requirements.txt


## streamlit tool for record:

  streamlit run scripts\pages.py

  Train 30 sec == > this record will go to train_demo Dir

  Test 5 sec ==> this record will go to test_demo Dir 



# After install requirements and record 30 sec Train and 5 sec Test

## steps to run this project:
  1- make features extraction:

      python scripts\pyannote.py

  2- train logistic regression model:

      python scripts\logistic_model.py

  2- run main file for inference:

      python scripts\main.py