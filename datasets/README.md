Traning dataset can be downloaded [here](https://drive.google.com/file/d/1JlZpgM5uBZI-xeayv-AnXnW_OQjzaUYP/view?usp=sharing).

## How were the training data prepared
The training data have been processed through following steps:

  1. Customer reviews from e-commerce websites were crawled. The crawled data include **reviews** (text) and **rating** (integers 1-5 corresponding to 1-5 rating stars).
  
  2. The **rating** data were converted into 3 class labels:
  
    1-2 stars => class label 0 (negative) 
    4-5 stars => class label 1 (positive) 
    3 stars   => class label 2 (neutral)
   
  3. The **class labels** were then manually reassigned. 

The dataset is saved in *utf-8 csv* format with 2 columns: **label**, **review**. There are about 100.000 reviews in the training data.

Thank *Trần Gia Bảo, Trần Thị Tâm Nguyên, Hoàng Thị Cẩm Tú,* and *Uông Thị Thanh Thủy* for helping me prepare this data.

## How to use your own data with `train.py`
Provide a csv file with 2 colunms: **label**, **review** as described above. Note that the class labels must be **0, 1, 2** as described in step 2.

**NOTE:** If you have a lot of data (e.g., >100.000 reviews) with relatively accurate rating, then you may **skip Step 3**. The model won't be affected much by several hundreds of wrong rating reviews. (However, the better data of course gives better model, as a well-known data science quote: *Garbage in, garbage out*).




