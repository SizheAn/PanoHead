# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:21:44 2022

@author: sizhe-admin
"""

from icrawler.builtin import BaiduImageCrawler 
from icrawler.builtin import BingImageCrawler 
from icrawler.builtin import GoogleImageCrawler 


both = [
'Buzz Cut',
'Short hair',
'Long Hair',
'fade cut',
'Crew cut',
'Flattop',
'Bowl cut',
'Undercut',
'Mohawk',
'Curtained hair',
'Slicked back',
'Bangs',
'Big hair',
'Jheri curl',
'spiky',
'Fauxhawk',
'Waves haircut',
'Braid',
'Dreadlocks',
'Perm',
]

boy = [
'Butch cut',
'Ivy League haircut',
'Caesar cut',
'High and tight',
'Jewfro',
'Man Bun',
'Chonmage',
'Comb over',
'Ducktail',
'Frosty Tips Hair',
'Mullet',
'Pompadour',
'Ponyhawk',
'Quiff',
'Shape Up',
]

female = [
'Bob cut',
'Pixie cut',
'bouffant hair',
'Bun',
'Fringe hairstyle',
'Pageboy hairstyle ',
'Chignon hairstyle',
'Crown braid',
'Feathered hair',
'Beehive hair',
'French braid',
'French twist',
'Hime cut',
'Layered hair',
'Lob',
'Marcel waves',
'Pigtail hair',
'Ponytail',
'Shag cut',
'Finger waves',
'Fishtail hair',
'Highlights',
'side swept',
'Updo',
'Asymmetric cut',
'Blunt Cut',
]

addtional = ['bald']

list_word = []

for list in [both, boy, female]:
# for list in [addtional]:
    for item in list:
        list_word.append(item)
        list_word.append(item + ' back')
        list_word.append(item + ' side view')

bing_filters = dict(
    size='large', type = 'photo')
    
bing_ex_filters = dict(
    size='extralarge', type = 'photo')
google_filters = dict(
    size='large', type = 'photo')
max_num = 5000

for word in list_word:
    # bing crawl
    # save directory
    bing_storage = {'root_dir': 'hairset_photo/bing/'+word}


    bing_crawler = BingImageCrawler(parser_threads=8,
                                    downloader_threads=8,
                                    storage=bing_storage)
    # Starting, keywords+amount
    bing_crawler.crawl(keyword=word,filters = bing_filters,
                       max_num=max_num)

    # bing crawl extra large
    # save directory
    bing_ex_storage = {'root_dir': 'hairset_photo/bingex/'+word}

    bing_ex_crawler = BingImageCrawler(parser_threads=8,
                                    downloader_threads=8,
                                    storage=bing_ex_storage)
    # Starting, keywords+amount
    bing_ex_crawler.crawl(keyword=word,filters = bing_ex_filters,
                       max_num=max_num)
                       
    # baidu crawl
    # baidu_storage = {'root_dir': 'hairset_photo/baidu/' + word}
    # baidu_crawler = BaiduImageCrawler(parser_threads=4,
    #                                   downloader_threads=4,
    #                                   storage=baidu_storage)
    # baidu_crawler.crawl(keyword=word,
    #                     max_num=max_num)


    # google crawl
    google_storage = {'root_dir': 'hairset_photo/google/' + word}
    google_crawler = GoogleImageCrawler(parser_threads=8,
                                       downloader_threads=8,
                                       storage=google_storage)
    google_crawler.crawl(keyword=word,filters = google_filters,
                         max_num=max_num)

