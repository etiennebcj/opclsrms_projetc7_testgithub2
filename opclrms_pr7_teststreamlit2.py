import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import shap
from sklearn.cluster import KMeans
from zipfile import ZipFile


# st.title('Credit Allocation Application')


# <-- def main() :

@st.cache
def load_data():
	
    z = ZipFile('X_data_rfecv_15000.zip') # OK size data < 200 MB
    data = pd.read_csv(z.open('X_data_rfecv_15000.csv')) # data = pd.read_csv(z.open('X_data_rfecv_32.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    
    z = ZipFile('X_data_rfecv_5000.zip')
    sample = pd.read_csv(z.open('X_data_rfecv_5000.csv')) 
    
    description = pd.read_csv('HomeCredit_columns_description.csv', 
    				usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
    				
    target = data.iloc[:, -1:]
    
    return data, sample, target, description

# data = load_data() # --> for displaying purpose
# st.subheader('data')
# st.write(data)


def load_model():
        '''loading the trained model'''
        pickle_in = open('LGBMClassifier_custom_score.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


@st.cache(allow_output_mutation=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn


@st.cache
def load_gen_info(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]


    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets
    
    
def client_identity(data, id):
    data_client = data[data.index == int(id)]
    return data_client    


@st.cache
def load_age_client(data):
    data_age = round((data["DAYS_BIRTH"]/365), 2)
    return data_age


@st.cache
def load_income_client(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income


@st.cache
def load_prediction(sample, id, clf):
    X=sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score


@st.cache
def load_kmeans(sample, id, mdl):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)


@st.cache
def knn_training(sample):
    knn = KMeans(n_clusters=2).fit(sample)
    return knn 


# Loading data
data, sample, target, description = load_data()
id_client = sample.index.values
clf = load_model()



#******************************************
# MAIN
#******************************************

# Title display
html_temp = """
<div style="background-color: LightSeaGreen; padding:5px; border-radius:10px">
	<h1 style="color: white; text-align:center">Credit Allocation Dashboard</h1>
</div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

# Customer ID selection
# st.sidebar.header('General Informations')

# Loading selectbox
chk_id = st.sidebar.selectbox('Client ID', id_client)

# Loading general informations
nb_credits, rev_moy, credits_moy, targets = load_gen_info(data)



#*******************************************
# Displaying informations on the sidebar
#*******************************************

# Number of loans for clients in study
st.sidebar.markdown("<u>Total number of loans in our study :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Average income
st.sidebar.markdown("<u>Average income ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u>Average loan amount ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)


# PieChart
fig, ax = plt.subplots(figsize=(5,5))
plt.pie(targets, explode=[0, 0.5], labels=['Reimbursed', 'Defaulted'], autopct='%1.1f%%', startangle=45)
st.sidebar.pyplot(fig)


# Barchart test
# ax, fig = plt.subplots(figsize=(5,5)) 
# ax = sn.countplot(data=target)
# ax.set_title('Credit allowance repartition')
# st.sidebar.pyplot(fig)


# Barchart test 2
# st.sidebar.bar_chart(target)



#******************************************
# MAIN -- suite
#******************************************

# Display Customer ID from Sidebar
st.write('Customer ID selected :', chk_id)

# Displaying customer information : gender, age, family status, Nb of hildren etc.
st.subheader('Informations')

if st.checkbox("Enable (Disable) customer summary"):
   infos_client = client_identity(data, chk_id)
   st.write("Gender : ", infos_client["CODE_GENDER"].values[0])
   st.write("Age : {:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
   st.write("Family status : ", infos_client["NAME_FAMILY_STATUS"].values[0])
   st.write("Number of children : {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
   
   # Age distribution plot
   data_age = load_age_population(data)
   fig, ax = plt.subplots(figsize=(10, 5))
   sn.histplot(data_age, edgecolor = 'b', color="darkorange", bins=15)
   ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
   ax.set(title='Customer age', xlabel='Age (Year)', ylabel='')
   st.pyplot(fig)
   
   st.subheader("Income ($US)")
   st.write("Income total : {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
   st.write("Credit amount : {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
   st.write("Credit annuities : {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
   st.write("Amount of property for credit : {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

   # Income distribution plot
   data_income = load_income_population(data)
   fig, ax = plt.subplots(figsize=(10, 5))
   sn.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'g', color="orange", bins=10)
   ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
   ax.set(title='Customer income', xlabel='Income ($US)', ylabel='')
   st.pyplot(fig)
   
   # Age vs Total income, interactive plot 
   data_at = data.reset_index(drop=False)
   data_at.DAYS_BIRTH = (data_at['DAYS_BIRTH']/365).round(1)
   fig, ax = plt.subplots(figsize=(9,9))
   fig = px.scatter(data_at, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                     size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                     hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])
                     
   fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Age vs Total income", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=15, family='Verdana'), legend=dict(y=1.1, orientation='h'))

   fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
   fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                    title="Age", title_font=dict(size=18, family='Verdana'))
   fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                    title="Total income", title_font=dict(size=18, family='Verdana'))                 
   
   st.pyplot(fig)  
   # st.plotly_chart(fig)                

else:
  st.markdown("<i>…</i>", unsafe_allow_html=True)


# Customer solvability display
st.header("Customer report")
prediction = load_prediction(sample, chk_id, clf)
st.write("Default probability : {:.0f} %".format(round(float(prediction)*100, 2)))

st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
st.write(client_identity(data, chk_id))    
    

#Feature importance / description
if st.checkbox("Show (Hide) customer #{:.0f} feature importance".format(chk_id)):
   shap.initjs()
   X = sample.iloc[:, :-1]
   X = X[X.index == chk_id]
   number = st.slider("Pick a number of features", 0, 20, 5)

   fig, ax = plt.subplots(figsize=(9,9))
   explainer = shap.TreeExplainer(load_model())
   shap_values = explainer.shap_values(X)
   shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
   st.pyplot(fig)
        
   if st.checkbox("Select feature for desciption") :
      list_features = description.index.to_list()
      feature = st.selectbox('Feature checklist', list_features)
      st.table(description.loc[description.index == feature][:1])
   
else:
    st.markdown("<i>…</i>", unsafe_allow_html=True)
    
    
# Similar customer to the one selected
neighbors_nearest = st.checkbox("Show (Hide) similar customer")

if neighbors_nearest:
   knn = load_knn(sample)
   st.markdown("<u>10 closest customers to the selected one :</u>", unsafe_allow_html=True)
   st.dataframe(load_kmeans(sample, chk_id, knn))
   st.markdown("<i>Target 1 = Customer high default probability</i>", unsafe_allow_html=True)
   
else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)    
    

    

# <-- if __name__ == '__main__':
    #main()























