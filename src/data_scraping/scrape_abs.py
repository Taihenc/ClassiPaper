from selenium import webdriver
from selenium.webdriver.common.by import By
from termcolor import colored
from lxml import html
import requests
import hashlib
import json
import os


cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

src_dir = 'data/scrape'
abs_src_dir = f'{src_dir}/abstracts'

abs_dst_dir = f'{src_dir}/abstracts_extra'
if not os.path.exists(abs_dst_dir):
    os.mkdir(abs_dst_dir)


# for experimenting
def get_page_content(page_url, fromcache=True):
    page_url_hash = hashlib.sha256(page_url.encode()).hexdigest()
    filepath = f'{cache_dir}/{page_url_hash}'
    if not os.path.exists(filepath) or not fromcache:
        print(colored(f'Fetching "{page_url}"...', 'yellow'))
        browser = webdriver.Firefox()
        browser.get(page_url)
        content = browser.page_source
        with open(filepath, 'w') as f:
            f.write(content)
    else:
        with open(filepath, 'r') as f:
            content = f.read()
    return content


# for experimenting
def test_lxml():
    page_urls = get_page_urls()
    content = get_page_content(page_urls[0])

    tree = html.fromstring(content)
    elm_authkeywords = tree.xpath('//section[@id="authorKeywords"]/span')
    keywords = list()
    for e in elm_authkeywords:
        keywords.append({ "@_fa": "true", '$': e.text })

    authkeywords = { "authkeywords": { "author-keyword": keywords }}
    print(json.dumps(authkeywords, indent=4))


# for experimenting
def selenium_test(url):
    try:
        browser = webdriver.Firefox()
        browser.get(url)

        author_keywords = browser.find_elements(By.ID, 'authorKeywords')
        abstract_section = browser.find_elements(By.ID, 'abstractSection')

        keyword_elms = author_keywords[0].find_elements(By.TAG_NAME, 'span')
        keywords = list()
        for e in keyword_elms:
            keywords.append(e.text)
        abstract = abstract_section[0].find_elements(By.TAG_NAME, 'p')[0].text
    except Exception as e:
        print(e)
    finally:
        browser.close()


# Scrape and add the abstract page and author keywords to dst_dict
# browser = a selenium webdriver
# url = url
# dst_dict = dict to be edited
#   - will change dst_dict['authkeywords']
#   - will change dst_dict['item']
def scrape(browser, url, dst_dict):
    browser.get(url)

    page_url_hash = hashlib.sha256(url.encode()).hexdigest()
    filepath = f'{cache_dir}/{page_url_hash}'
    if not os.path.exists(filepath):
        content = browser.page_source
        with open(filepath, 'w') as f:
            f.write(content)

    author_keyword_elms = browser.find_elements(By.ID, 'authorKeywords')
    author_keyword = None
    if author_keyword_elms:
        keyword_elms = author_keyword_elms[0].find_elements(By.TAG_NAME, 'span')
        if keyword_elms:
            keywords = list()
            for e in keyword_elms:
                keywords.append({'@_fa': 'true', '$': e.text})
            author_keyword = { 'author-keyword': keywords }

    abstract_section_elms = browser.find_elements(By.ID, 'abstractSection')
    abstract_text = None
    if abstract_section_elms:
        p_elms = abstract_section_elms[0].find_elements(By.TAG_NAME, 'p')
        if p_elms:
            if len(p_elms) != 1:
                print(colored('Unexpected paragraph!:', 'red'), f'"{url}"')
                print(colored('will merge them into one.', 'red'))
                abstract_text = p_elms[0].text
                for e in p_elms[1:]:
                    abstract_text += ' ' + e.text
            elif p_elms[0].text != '[No abstract available]':
                abstract_text = p_elms[0].text

    dst_dict['authkeywords'] = author_keyword
    dst_dict['item'] = {'bibrecord': {'head': {'abstracts': abstract_text}}}


# Scrape abstract text and author keywords
# from all abstracts in './data/scrape/abstracts'
# put results in './data/scrape/abstracts_extra'
def scrape_all():
    srcs = os.listdir(abs_src_dir)
    browser = webdriver.Firefox()
    try:
        for src in srcs:
            with open(f'{abs_src_dir}/{src}', 'r') as f:
                src_json = json.load(f)

            content = src_json['abstracts-retrieval-response']
            link_infos = content['coredata']['link']
            for i in link_infos:
                if i['@rel'] == 'scopus':
                    url = i['@href']
                    break
            if not url:
                print(colored('No page url found.', 'red'))
                continue

            scrape(browser, url, content)

            with open(f'{abs_dst_dir}/{src}', 'w') as f:
                json.dump(src_json, f, indent=2)
    except Exception as e:
        print(e)
    finally:
        browser.close()


def main():
    scrape_all()


if __name__ == '__main__':
    main()
