from urllib.parse import urlparse, parse_qs, urljoin
from termcolor import colored
import os
import json
import requests

g_testing = False

# possible search api view values
STANDARD = 'STANDARD' # (default)
COMPLETE = 'COMPLETE' # auth err

# possible abstract api view values
META = 'META' # (default)
META_ABS = 'META_ABS' # auth err
REF = 'REF' # auth err
ENTITLED = 'ENTITLED' # we're not entitled duh
FULL = 'FULL' # auth err


g_api_url = 'https://api.elsevier.com/content'
g_search_url = f'{g_api_url}/search/scopus'
g_abstract_url = f'{g_api_url}/abstract/scopus_id'

with open('api_key', 'r') as f:
    g_api_key = f.read().splitlines()[0]

if g_testing:
    dst_dir = 'data/test/scrape'
    indent = 4
else:
    dst_dir = 'data/scrape'
    indent = 2

search_dst_dir = f'{dst_dir}/search'
abs_dst_dir = f'{dst_dir}/abs'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

if not os.path.exists(search_dst_dir):
    os.makedirs(search_dst_dir)

if not os.path.exists(abs_dst_dir):
    os.makedirs(abs_dst_dir)


# util
def create_file(filepath, write_callback, overwrite=True):
    path = filepath
    if os.path.exists(path):
        if not overwrite:
            name, ext = os.path.splitext(path)
            i = 1
            path = name + f'_<{i}>' + ext
            while os.path.exists(path):
                i += 1
                path = name + f'_<{i}>' + ext
        else:
            print(colored('overwriting:', 'red'), path)

    print(colored('writing to file:', 'green'), path)
    with open(path, 'w') as f:
        write_callback(f)


# for experimenting
# max count = 25
def fetch_search_api(query, view=None, count=None):
    params = {
        'apiKey': g_api_key,
        'count' : count,
        'view' : view,
        'query': query,
    }

    headers = { 'Accept': 'application/json' }
    res = requests.get(g_search_url, params=params, headers=headers)
    print(res.url)
    if res.status_code != 200:
        print(res.json())
        raise Exception('status code not 200')

    res_json = res.json()
    create_file(f'./{search_dst_dir}/search_{view}_{count}-{'_'.join(query.split(' '))}.json', lambda f: json.dump(res_json, f, indent=indent))
    return res_json


# For fetch_searches()
# Fetch $next_url, put json response in $search_dst_dir
# returns json response
def fetch_search_next(next_url):
    print(colored('next:', 'green'), next_url)
    parsed_url = urlparse(next_url)
    url_query = parse_qs(parsed_url.query)

    # strips params
    next_url = urljoin(next_url, parsed_url.path)

    url_query['apiKey'] = g_api_key
    query = url_query['query'][0]
    view = url_query['view'][0]
    start = url_query['start'][0]
    count = url_query['count'][0]

    headers = { 'Accept': 'application/json' }
    res = requests.get(next_url, params=url_query, headers=headers)
    print(res.url)
    if res.status_code != 200:
        print(res.json())
        raise Exception('status code not 200')

    res_json = res.json()
    create_file(f'./{search_dst_dir}/search_{view}_{count}-{'_'.join(query.split(' '))}_{start}.json', lambda f: json.dump(res_json, f, indent=indent))
    return res_json


# Fetch $search_count entries after $first_j
#
# first_j = json from a Scopus Search API response
# search_count = the number of entries (papers) to be searched
# start_count = the starting number of entries (papers) counted
# desc = just for tracking stuff
def fetch_searches(first_j, start_count=0, search_count=1000, desc=None):
    first_j = first_j['search-results']
    entry = first_j['entry']
    link = first_j['link']

    count = start_count
    while count < search_count:
        next_url = None
        for l in link:
            ref = l['@ref']
            if ref == 'next':
                next_url = l['@href']
        if not next_url:
            break

        j = fetch_search_next(next_url)
        j = j['search-results']
        entry = j['entry']
        link = j['link']
        count += len(entry)

    with open(f'{dst_dir}/search_count.txt', 'a') as f:
        print(f"{desc} : {count}", file=f)

    print('search fetch done.')


def fetch_searches_from_start(query='PUBYEAR = 2024'):
    first_j = fetch_search_api(query, view=STANDARD, count=25)
    start_count = len(first_j['search-results']['entry'])
    fetch_searches(first_j, desc=f"'{query}'", start_count=start_count)


# For fetch_abs_from_url()
# Fetch Abstract Retrieval API from scopus_id, with $view
# NOTE: views other than 'META' have auth errors
def fetch_abs_from_scopus_id(scopus_id, view=None):
    params = {
        'apiKey': g_api_key,
        'view' : view,
    }

    headers = { 'Accept': 'application/json' }
    res = requests.get(f'{g_abstract_url}/{scopus_id}', params=params, headers=headers)
    print(res.url)
    if res.status_code != 200:
        print(res.json())
        raise Exception('status code not 200')

    create_file(f'./{abs_dst_dir}/{scopus_id}_{view}.json', lambda f: json.dump(res.json(), f, indent=indent))


# For fetch_abs_from_searches()
# Fetch Abstract Retrieval API from $abs_url, with $view
def fetch_abs_from_url(abs_url, view=None):
    print(abs_url)
    parsed_url = urlparse(abs_url)
    if parsed_url.query:
        raise Exception('abstract url has unexpected query params!')

    # WARN: assuming it is scopus_id at the back and no orphaned / at the back
    scopus_id = abs_url.split('/')[-1]
    fetch_abs_from_scopus_id(scopus_id, view=view)


# Fetch Abstract Retrieval API from entries in Scopus Search API responses
# The responses are from $search_dst_dir (should be in './data/scrape/search')
def fetch_abs_from_searches():
    searches = os.listdir(f'{search_dst_dir}')
    count = 0
    for search in searches:
        with open(f'{search_dst_dir}/{search}', 'r') as f:
            j = json.load(f)
        j = j['search-results']
        entry = j['entry']
        for e in entry:
            url = e['prism:url']
            fetch_abs_from_url(url)
            count += 1

    with open(f'{dst_dir}/abs_count.txt', 'a') as f:
        print(count, file=f)


def main():
    fetch_searches_from_start()


if __name__ == '__main__':
    main()
