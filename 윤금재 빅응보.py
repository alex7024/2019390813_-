#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. 라이브러리
import os
import pandas as pd
import pefile
from tqdm import tqdm

# 2. 경로 설정
label_csv_path = '/Users/yoonchanghoon/Desktop/빅데이터 중간 프로젝트/train_data_label.csv'
exe_folder_path = '/Users/yoonchanghoon/Desktop/빅데이터 중간 프로젝트/train_dataset/'

# 3. 라벨 로드
label_df = pd.read_csv(label_csv_path)

# 4. 다양한 후보 feature 추출 함수 (아직 선정 아님)
def extract_candidate_features(filepath):
    try:
        pe = pefile.PE(filepath)
        return {
            'SizeOfCode': pe.OPTIONAL_HEADER.SizeOfCode,
            'SizeOfInitializedData': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'SizeOfUninitializedData': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            'AddressOfEntryPoint': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'CheckSum': pe.OPTIONAL_HEADER.CheckSum,
            'NumberOfSections': pe.FILE_HEADER.NumberOfSections,
            'Characteristics': pe.FILE_HEADER.Characteristics,
            'Subsystem': pe.OPTIONAL_HEADER.Subsystem
        }
    except Exception:
        return {
            'SizeOfCode': None,
            'SizeOfInitializedData': None,
            'SizeOfUninitializedData': None,
            'AddressOfEntryPoint': None,
            'CheckSum': None,
            'NumberOfSections': None,
            'Characteristics': None,
            'Subsystem': None
        }

# 5. Feature 추출
feature_rows = []
for filename in tqdm(label_df['filename'], desc='Extracting PE features'):
    full_path = os.path.join(exe_folder_path, filename)
    row = extract_candidate_features(full_path)
    row['filename'] = filename
    feature_rows.append(row)

# 6. 병합 및 정리
feature_df = pd.DataFrame(feature_rows)
merged_df = pd.merge(label_df, feature_df, on='filename', how='left')
merged_df = merged_df.dropna()
merged_df['label'] = merged_df['label'].astype(int)

# 7. 저장
merged_df.to_csv('/Users/yoonchanghoon/Desktop/빅데이터 중간 프로젝트/eda_candidates.csv', index=False)

print("✅ 다양한 후보 feature 저장 완료! 이제 EDA로 선별 시작하면 돼.")


# In[2]:


# 1. 필요한 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 데이터 불러오기
df = pd.read_csv('/Users/yoonchanghoon/Desktop/빅데이터 중간 프로젝트/eda_candidates.csv')

# 3. 기본 정보 확인
print(df.info())
print(df.describe())
print(df['label'].value_counts())

# 4. feature별 분포 시각화 (label별로 구분)
key_features = [
    'SizeOfCode',
    'SizeOfInitializedData',
    'SizeOfUninitializedData',
    'AddressOfEntryPoint',
    'CheckSum',
    'NumberOfSections',
    'Characteristics',
    'Subsystem'
]

for col in key_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y=col, data=df, palette='husl')
    plt.title(f'{col} by Label')
    plt.xlabel('Threat Type (Label)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# In[3]:


df.groupby('label')[['SizeOfCode', 'AddressOfEntryPoint', 'NumberOfSections', 'CheckSum']].agg(['mean', 'std'])


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = df[['SizeOfCode', 'AddressOfEntryPoint', 'NumberOfSections', 'CheckSum']].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation")
plt.show()


# In[5]:


sns.scatterplot(data=df, x='SizeOfCode', y='AddressOfEntryPoint', hue='label', palette='husl')
plt.title("SizeOfCode vs AddressOfEntryPoint (by label)")
plt.show()


# In[6]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = df[['SizeOfCode', 'AddressOfEntryPoint', 'NumberOfSections', 'CheckSum']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='barh')
plt.title("Feature Importance (DecisionTree)")
plt.show()


# In[9]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

X = df[['AddressOfEntryPoint', 'CheckSum', 'NumberOfSections']]
X_scaled = StandardScaler().fit_transform(X)

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

df['tsne-1'] = X_tsne[:, 0]
df['tsne-2'] = X_tsne[:, 1]

sns.scatterplot(x='tsne-1', y='tsne-2', hue='label', data=df, palette='husl')
plt.title("t-SNE Visualization of PE Features")
plt.show()


# In[10]:


sns.countplot(x='label', data=df, palette='husl')
plt.title('Label Distribution')
plt.show()


# In[ ]:




