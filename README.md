# A Study on the Quality and Usability of Privacy-Friendly Synthetic Data

## 📌 Project Overview

There is currently a **huge demand for data** in industry. Since this data often contains **sensitive personal information**, it cannot be shared freely. In addition to existing **anonymization techniques**, we are currently seeing a significant movement in both **academia and industry** towards the use of **synthetic data**. The reasoning is that, since the data is **synthetic (artificial)**, it contains **no information about individuals**, and thus there are no privacy concerns.

Nevertheless, in reality, **various papers** have already pointed out that **synthetic data can leak information** about individuals in the original dataset on which the synthetic dataset is based. Unlike traditional **anonymization techniques**, it is also **very difficult to quantify the privacy level** of a synthetic dataset. Therefore, we propose a **hybrid solution** where the data is **(partially) anonymized** before being used to generate a synthetic dataset. In this way, **minimal guarantees** can be provided about the **privacy properties**.

The goal of this master’s thesis is to **analyze the usefulness** of the data obtained by applying this **hybrid solution**. To achieve this, a **test setup** will be created that allows a **wide range of experiments** to be easily conducted. The **usability of different alternatives**—**original data, synthetic data, anonymized data, and data obtained through the application of the hybrid solution**—will be compared by using them in **realistic use cases** (e.g., building an ML model). The results should provide an idea of the **practical usefulness** of our **hybrid solution**.


## 🎯 Objectives

- **Analyze** the performance of machine learning models when trained on anonymized data versus synthetic data, considering both **privacy** and **utility**.

- **Develop a hybrid approach** that integrates **partial anonymization with synthetic data generation** to enhance **privacy protection** while maintaining the **utility of the resulting data**.

- **Develop a reusable pipeline** for systematically testing and evaluating datasets (**original, anonymized, synthetic, and hybrid**) with **minimal adjustments**, enabling consistent comparisons of model performance and privacy metrics across different configurations.

- **Identify the best configuration** for the hybrid model to achieve an **optimal balance between privacy and usability**.

## Contributors

- **Dele Ayeni** - [GitHub Profile](https://github.com/deleayeni)
- **Isabelle De Andrade Costa** - [GitHub Profile](https://github.com/isabelledeac)
