# Numpy and pandas by default assume a narrow screen - this fixes that
from fastai.vision.all import *
from nbdev.showdoc import *
from ipywidgets import widgets
from pandas.api.types import CategoricalDtype

import matplotlib as mpl
# mpl.rcParams['figure.dpi']= 200
mpl.rcParams['savefig.dpi']= 200
mpl.rcParams['font.size']=12

set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pd.set_option('display.max_columns',999)
np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

import graphviz
def gv(s): return graphviz.Source('digraph G{ rankdir="LR"' + s + '; }')

def get_image_files_sorted(path, recurse=True, folders=None): return get_image_files(path, recurse, folders).sorted()


# +
# pip install azure-cognitiveservices-search-imagesearch

from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
from msrest.authentication import CognitiveServicesCredentials as auth

def search_images_bing(key, term, min_sz=128):
    client = api('https://api.cognitive.microsoft.com', auth(key))
    return L(client.images.search(query=term, count=150, min_height=min_sz, min_width=min_sz).value)

def search_images_google(key, term, min_sz=128, start=1, num=10,exp=False, search_engine="1255cf7ee95a87668"):
    results=L()
    custom_search_engine_id = search_engine #backup: "da50e89486da1614e"
    search_url = "https://www.googleapis.com/customsearch/v1"
    while start<100:
        params = {"q":term,"searchType":"image","safe":"off","imgType":"photo","num":num,"start":start,"cx":custom_search_engine_id,"key":key}
        response = requests.get(search_url, params=params)
        try:
            response.raise_for_status()
        except Exception as e:
            if custom_search_engine_id == "1255cf7ee95a87668":
                custom_search_engine_id = "da50e89486da1614e"
                key = "AIzaSyD-mpwppP58hJugiuFEyRu3RdqG2ObDvlQ"
                continue
            else:
                raise e
        search_results = response.json()
        try:
            results += L(search_results["items"])
        except KeyError:
            continue
        start+=num
    
    if exp==True: #double the search by appending "porn" onto the term
        term = term+" porn"
        start=1
        num=10
        while start<100:
            params = {"q":term,"searchType":"image","safe":"off","imgType":"photo","num":num,"start":start,"cx":custom_search_engine_id,"key":key}
            response = requests.get(search_url, params=params)
            try:
                response.raise_for_status()
            except Exception as e:
                if custom_search_engine_id == "1255cf7ee95a87668":
                    custom_search_engine_id = "da50e89486da1614e"
                    key = "AIzaSyD-mpwppP58hJugiuFEyRu3RdqG2ObDvlQ"
                    continue
                else:
                    raise e
            search_results = response.json()
            try:
                results += L(search_results["items"])
            except KeyError:
                continue
            start+=num
    
    return results
    
#def search_images_bing(key, term, min_siz=128):
    #headers = {"Ocp-Apim-Subscription-Key" : key}
    #search_url = 'https://api.bing.microsoft.com/v7.0/images/search'
    #params  = {"q": term, "imageType": "photo","count":150, "safeSearch":"Off","minWidth":min_sz,"minHeight":min_sz}
    #response = requests.get(search_url, headers=headers, params=params)
    #response.raise_for_status()
    #search_results = response.json()
    ##thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]] #Microsoft boilerplate
    #return L(search_results["value"])


# -

def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)

# +
from sklearn.tree import export_graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))


# +
from scipy.cluster import hierarchy as hc

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()
