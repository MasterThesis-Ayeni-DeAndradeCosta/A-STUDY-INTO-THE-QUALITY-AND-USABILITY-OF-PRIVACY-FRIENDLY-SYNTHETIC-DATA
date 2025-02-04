# A Study on the Quality and Usability of Privacy-Friendly Synthetic Data

## 📌 Project Overview

There is a growing demand for data in Industry. However, sharing sensitive personal information is a major concern. **Synthetic data**, created artificially, offers a potential solution. However, studies have shown that **synthetic data can still leak information** about individuals.

Another potential solution is a **hybrid approach**: partially anonymizing data before generating synthetic data. This provides **minimal privacy guarantees**.

The goal of this master’s thesis is to analyze the **usefulness of synthetic data** by comparing it with original and anonymized data in **realistic use cases** (e.g., building an ML model). Following this analysis, **a hybrid solution will be proposed**. To evaluate this approach, a **test setup** will be created to conduct a **wide range of experiments** easily. The usability of different alternatives—**original data, synthetic data, anonymized data, and hybrid data**—will then be compared. The results should provide an idea of the **practical usefulness** of the hybrid solution.

---

## 🎯 Objectives

- 📌 **Analyze** the performance of machine learning models when trained on anonymized data versus synthetic data, considering both **privacy** and **utility**.

- 🛠️ **Develop a hybrid approach** that integrates **partial anonymization with synthetic data generation** to enhance **privacy protection** while maintaining the **utility of the resulting data**.

- ⚙️ **Develop a reusable pipeline** for systematically testing and evaluating datasets (**original, anonymized, synthetic, and hybrid**) with **minimal adjustments**, enabling consistent comparisons of model performance and privacy metrics across different configurations.

- 🌟 **Identify the best configuration** for the hybrid model to achieve an **optimal balance between privacy and usability**.
