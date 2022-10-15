CMPT 353 Project Sensors, Noise, and Walking
 

Required Libraries: 
`pip3 install numpy pandas scipy matplotlib sklearn seaborn`

Instructions:
Normal_classifier_model(Injury_classifier_model) is used to plug in an unknown file to test whether this
data file is predicted to be a normal person or a injuried person. But there
may be a serveral case in order to test the validity. For example,

Running the following command for the normal classifier model
`python3 normal_classifier_model.py hand_1`

Running the following command for the injury classifier model
`python3 injury_classifier_model.py injury`

All the name of file that you can add it: 
`  'female_1','female_2','female_3' , 'female_4' , 'female_5' , 'hand_1' , 'hand_2' , 'left_foot' , 'male_1'`
` 'male_2' , 'male_3' , 'male_4' , 'male_5', 'right_foot_1','right_foot_2','right_foot_3','injury','injury_1','injury_2'`

Output: 
If we plug an unknown file into normal_classifier_model or injury_classifier_model
 and the test result is[0] which represents normal, and plug the file into 
injury_classifier_model and the result is [0] which rpresents normal in 
injury_classifier, then the file has a high possibility from a normal person.
But if the file is shown [0] as normal person in normal_classifier_model and [1] 
as injury person in injury_classifier_model, then we conclude that this data file
is abnormal, so it does not meet our expectations. 


Output Example: 
All the line before next output is all "train score", "valid score", "injury situation" of input file which built that classifier.

The injury situation of input right_foot_1 is  0
The train socre of input right_foot_1 is  0.4507690567380864
The valid socre of input right_foot_1 is  0.46812181715431733
...
`The result of input file in [YOUR CLASSIFIER MODEL] classifier is (Note 1 means injury; 0 means not injury)  ['YOUR RESULT']`
