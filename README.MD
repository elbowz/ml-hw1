# Machine Learning: KNN using sklearn  
### more detailed information in ITALIAN: [REPORT.MD](REPORT.MD)  

Comparing K Nearest Neighbor with different **k** and **metrics**.

## Utilized libs 
    
* **pandas**:  load and manipulation of **DataSet** 
* **sklearn**:  algo, stats and splitting of **TrainingSet**  
  
> **note**: see `main-from-scratch.py` + `lib.py` for a an implementation without external library

## Script usage  
  
```bash  
main.py [dataset] [test_size]  
```  
  
* **dataset**: [wine|breast-cancer|letters|poker] - select DataSet. Default is *wine*  
* **testset_size**: [0,1] - TestSet dimension  

## DataSets

The **DataSet** taken from [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/index.php) are: 
* [Wine](https://archive.ics.uci.edu/ml/datasets/Wine)
* [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
* [Letter Recognition](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
* [Poker Hand](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)