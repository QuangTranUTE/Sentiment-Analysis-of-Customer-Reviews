## How were the data prepared
The training data have been processed through following steps:

  1. Customer reviews from e-commerce websites were crawled. The crawled data include **reviews** (text) and **rating** (integers 1-5 corresponding to 1-5 rating stars).
  
  2. The **rating** were converted into 3 classes:
  
    Class 0 (negative): 1-2 stars
    Class 1 (positive): 4-5 stars
    Class 2 (neutral): 3 stars
   
  3. The **class labels** were then manually reassigned. 

The dataset is saved in *utf-8 csv* format with 2 columns: **label**, **review**. There are about 100.000 reviews in the training data.

Thank *Trần Gia Bảo, Trần Thị Tâm Nguyên, Hoàng Thị Cẩm Tú,* and *Uông Thị Thanh Thủy* for helping me prepare this data.




