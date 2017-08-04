import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")

from lxml import html
import requests
import os
import csv

from requests_testadapter import Resp

class LocalFileAdapter(requests.adapters.HTTPAdapter):
    def build_response_from_file(self, request):
        file_path = request.url[7:]
        with open(file_path, 'rb') as file:
            buff = bytearray(os.path.getsize(file_path))
            file.readinto(buff)
            resp = Resp(buff)
            r = self.build_response(request, resp)

            return r

    def send(self, request, stream=False, timeout=None,
             verify=True, cert=None, proxies=None):

        return self.build_response_from_file(request)

requests_session = requests.session()
requests_session.mount('file://', LocalFileAdapter())

number = 1

with open('../Resource/addition.csv', 'w') as csvfile:
    fieldnames = ['no','id', 'content','main post','poster post','label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()

    for count in range(10):
        print("test",count)
        akun = []
        konten = []

        filename = 'addition'
        count += 1
        filename += `count`

        page = requests_session.get('file://../File HTML/' + filename+'.html')
        tree = html.fromstring(page.content)

        # for post in tree.xpath('//a[@class="_4zhc5 notranslate _iqaka"]/text()'):
        for post in tree.xpath('//a[@class="_4zhc5 notranslate _24641"]/text()'):
            akun.append(post)


        # for results in tree.xpath('.//ul[@class="_mo9iw _123ym"]/li'):
        for results in tree.xpath('.//ul[@class="_h3rdq"]/li'):
            konten.append(results.text_content().encode('utf-8'))

        for index in range(len(akun)):
            if index == 0:
                writer.writerow({'no':number, 'id':akun[index], 'content':konten[index][len(akun[index]):],'main post':'yes' , 'poster post':'yes', 'label':'x'})
                poster = akun[index]
            else:
                if akun[index] == poster:
                    writer.writerow({'no':number, 'id':akun[index], 'content':konten[index][len(akun[index]):],'main post':'no', 'poster post':'yes', 'label':'x'})
                else:
                    writer.writerow({'no':number, 'id':akun[index], 'content':konten[index][len(akun[index]):],'main post':'no', 'poster post':'no', 'label':'x'})
            number += 1
        
