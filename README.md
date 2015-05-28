## Scalable Machine Learning

###Prerequisite 
- numpy
- Pandas
- spyLearn
- Scikit-Learn- PIL
- matplotlib
- pyspark

###How to run
- Set up your own spark path
	- Export SPARK_HOME
	
			export $SPARK_HOME=YOURPATH/spark-1.1.1-bin-hadoop2.4
		
	- Export Pyspark Path
	
			export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
		
- Place data file under the same directory with the scripts
	
- Run PCA scripts

	- python pca.py

###Sample result plot

![PCA result](https://raw.githubusercontent.com/Nero-Hu/Bigdata_Project/master/Final%20Report/tex/pca.png)
