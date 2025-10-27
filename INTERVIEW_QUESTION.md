# 🌳 Interview Questions — Decision Trees & Ensemble Learning

### What You'll Learn
- Decision Tree working  
- Ensemble learning (Bagging & Random Forest)  
- Feature importance  
- Overfitting control  

---

### **How does a Decision Tree work?**
A **Decision Tree** splits data into smaller groups based on feature values using questions like:  
> “Is age > 45?” or “Cholesterol ≤ 240?”

At each node, it chooses the **best feature** that separates classes most clearly (based on **information gain** or **Gini index**).  
The process continues until a **stopping condition** (e.g., max depth or minimum samples per leaf) is met.

---

### **What is Entropy and Information Gain?**

- **Entropy** measures the impurity or disorder of data.  
  \[
  Entropy = -\sum p_i \log_2(p_i)
  \]  
  where \(p_i\) is the probability of each class.

- **Information Gain** measures how much entropy decreases after a split:  
  \[
  Information\ Gain = Entropy(parent) - \text{Weighted Avg of Entropy(children)}
  \]

The feature with **higher information gain** is chosen for splitting.

---

### **How is Random Forest better than a single tree?**
- A **single decision tree** can easily overfit.  
- A **Random Forest** builds many trees using random subsets of data and features.  
- It averages or votes across trees → reduces **variance** and increases **accuracy**.

---

### **What is Overfitting and how do you prevent it?**

**Overfitting** happens when a model memorizes training data instead of learning patterns.

**Prevention:**
- Limit tree **max_depth**
- Use **min_samples_split** or **min_samples_leaf**
- Apply **pruning**
- Use **Random Forest** or **Cross-validation**

---

### **What is Bagging?**

**Bagging (Bootstrap Aggregating)** is an ensemble method where:
- Multiple models (e.g., trees) are trained on **random subsets** of data (with replacement).  
- Their predictions are **averaged** (regression) or **voted** (classification).
  
 Random Forest = Bagging + Random Feature Selection

---

### **How do you visualize a Decision Tree?**

Using **Graphviz** or **Matplotlib**:

```python
from sklearn.tree import plot_tree
plot_tree(tree_model, filled=True, feature_names=X.columns)

```python
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(model, out_file=None, feature_names=X.columns,
                           class_names=['No', 'Yes'], filled=True, rounded=True)
graphviz.Source(dot_data)

### **7. How do you interpret Feature Importance?**

Feature importance shows which features contribute most to the model’s decisions.
In tree-based models, it is calculated by how much each feature decreases impurity (e.g., Gini index or entropy) across all splits.

### **8. What are the Pros and Cons of Random Forests?**

Random Forests have several advantages and disadvantages. They are highly accurate and robust because they combine multiple decision trees, which reduces overfitting and improves generalization. They can handle large datasets, missing values, and complex non-linear relationships effectively. However, they also have some drawbacks — training can be slower compared to a single tree, the model is harder to interpret due to its ensemble nature, and it consumes more memory. Additionally, it may require tuning of hyperparameters like the number of trees or maximum depth to achieve optimal performance.



