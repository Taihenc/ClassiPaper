from termcolor import colored
import os
import json

dst_dir = 'data/scrape'
search_dst_dir = f'{dst_dir}/search'

searches = os.listdir(f'{search_dst_dir}')

dupe_id_set = set()
id_set = set()
id_list = list()
for search in searches:
    with open(f'{search_dst_dir}/{search}', 'r') as f:
        content = json.load(f)

    content = content['search-results']
    entry = content['entry']
    for e in entry:
        id = e['dc:identifier'].split(':')[1]
        assert(id)
        assert(id != 'SCOPUS_ID')
        id_list.append(id)

        if id in id_set:
            print(colored(id, 'red'))
            dupe_id_set.add(id)
            continue

        id_set.add(id)

print('ids that have dupes count:', len(dupe_id_set))
print('ids count (unique):', len(id_set))
print('ids count (count dupes):', len(id_list))
