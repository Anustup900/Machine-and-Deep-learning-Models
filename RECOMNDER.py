#Recommend similar jobs based on the jobs title, description
#Recommend jobs based on similar user profiles

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
apps = pd.read_csv('', delimiter='\t',encoding='utf-8')
user_history = pd.read_csv('', delimiter='\t',encoding='utf-8')
jobs = pd.read_csv('', delimiter='\t',encoding='utf-8', error_bad_lines=False)
users = pd.read_csv('' ,delimiter='\t',encoding='utf-8')
test_users = pd.read_csv('', delimiter='\t',encoding='utf-8')
apps_training = apps.loc[apps['Split'] == 'Train']
apps_testing = apps.loc[apps['Split'] == 'Test']
user_history_training = user_history.loc[user_history['Split'] =='Train']
user_history_training = user_history.loc[user_history['Split'] =='Train']
user_history_testing = user_history.loc[user_history['Split'] =='Test']
apps_training = apps.loc[apps['Split'] == 'Train']
apps_testing = apps.loc[apps['Split'] == 'Test']
users_training = users.loc[users['Split']=='Train']
users_testing = users.loc[users['Split']=='Test']
user_history_testing = user_history.loc[user_history['Split'] =='Test']
users_training = users.loc[users['Split']=='Train']
jobs.groupby(['City','State','Country']).size().reset_index(name='Locationwise')
jobs.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)
jobs_US = jobs.loc[jobs['Country']=='INDIA']
jobs_INDIA[['City','State','Country']]
jobs_INDIA.groupby(['City','State','Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False).head()  
State_wise_job_US = jobs_US.groupby(['State']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)
jobs_US.groupby(['City']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)
City_wise_location = jobs_US.groupby(['City']).size().reset_index(
    name='Locationwise').sort_values('Locationwise',ascending=False)
City_wise_location_th = City_wise_location.loc[City_wise_location['Locationwise']>=12]
users_training.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False).head()
user_training_US = users_training.loc[users_training['Country']=='INDIA']
user_training_US.groupby(['State']).size().reset_index(
    name='Locationwise_state').sort_values('Locationwise_state',ascending=False)
user_training_INDIA_state_wise = user_training_INDIA.groupby(['State']).size().reset_index(
    name='Locationwise_state').sort_values('Locationwise_state',ascending=False) 
user_training_INDIA_th = user_training_INDIA_state_wise.loc[user_training_INDIA_state_wise['Locationwise_state']>=12] 
user_training_INDIA.groupby(['City']).size().reset_index(
    name='Locationwise_city').sort_values('Locationwise_city',ascending=False)
user_training_INDIA_city_wise = user_training_INDIA.groupby(['City']).size().reset_index(
    name='Locationwise_city').sort_values('Locationwise_city',ascending=False)   
user_training_INDIA_City_th = user_training_INDIA_city_wise.loc[user_training_INDIA_city_wise['Locationwise_city']>=12]    
jobs_INDIA_base_line['Title'] = jobs_INDIA_base_line['Title'].fillna('')
jobs_INDIA_base_line['Description'] = jobs_INDIA_base_line['Description'].fillna('')


jobs_INDIA_base_line['Description'] = jobs_INDIA_base_line['Title'] + jobs_INDIA_base_line['Description']      
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(jobs_US_base_line['Description'])   
# http://scikit-learn.org/stable/modules/metrics.html#linear-kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
jobs_INDIA_base_line = jobs_INDIA_base_line.reset_index()
titles = jobs_INDIA_base_line['Title']
indices = pd.Series(jobs_INDIA_base_line.index, index=jobs_INDIA_base_line['Title'])
#indices.head(2)                     
def get_recommendations(title):
    idx = indices[title]
    #print (idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print (sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    job_indices = [i[0] for i in sim_scores]
    return titles.iloc[job_indices]
 get_recommendations('SAP Business Analyst / WM').head(10)
 get_recommendations('Security Engineer/Technical Lead').head(10)
 get_recommendations('Immediate Opening').head(10)
 get_recommendations('EXPERIENCED ROOFERS').head(10)
 user_based_approach_US = users_training.loc[users_training['Country']=='INDIA']                                                                
 user_based_approach = user_based_approach_US.iloc[0:10000,:]
 user_based_approach['DegreeType'] = user_based_approach['DegreeType'].fillna('')
user_based_approach['Major'] = user_based_approach['Major'].fillna('')
user_based_approach['TotalYearsExperience'] = str(user_based_approach['TotalYearsExperience'].fillna(''))

user_based_approach['DegreeType'] = user_based_approach['DegreeType'] + user_based_approach['Major'] + user_based_approach['TotalYearsExperience']
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(user_based_approach['DegreeType'])
# http://scikit-learn.org/stable/modules/metrics.html#linear-kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
user_based_approach = user_based_approach.reset_index()
userid = user_based_approach['UserID']
indices = pd.Series(user_based_approach.index, index=user_based_approach['UserID'])
#indices.head(2)
def get_recommendations_userwise(userid):
    idx = indices[userid]
    #print (idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print (sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    user_indices = [i[0] for i in sim_scores]
    #print (user_indices)
    return user_indices[0:11]
print ("-----Top 10 Similar users with userId: 123------")
get_recommendations_userwise(123)
def get_job_id(usrid_list):
    jobs_userwise = apps_training['UserID'].isin(usrid_list) #
    df1 = pd.DataFrame(data = apps_training[jobs_userwise], columns=['JobID'])
    joblist = df1['JobID'].tolist()
    Job_list = jobs['JobID'].isin(joblist) #[1083186, 516837, 507614, 754917, 686406, 1058896, 335132])
    df_temp = pd.DataFrame(data = jobs[Job_list], columns=['JobID','Title','Description','City','State'])
    return df_temp
 print ("-----Top 10 Similar users with userId: 47------")
get_recommendations_userwise(47)
get_job_id(get_recommendations_userwise(47))


                                    

                                                                                                                                                                                                                           
