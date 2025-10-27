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
