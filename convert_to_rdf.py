import xml.etree.ElementTree as ET
from multiprocessing.pool import Pool

import pandas as pd

from CNQuery import CNQuery

api_address = "8kboxx/query?"
filename = "retrogan/conceptnet-assertions-5.6.0.csv"
def xml_entry(start,end,relationship,index):
    '''
    <entry category="Airport" eid="Id1" size="1">
      <originaltripleset>
        <otriple>Aarhus_Airport | cityServed | "Aarhus, Denmark"@en</otriple>
      </originaltripleset>
      <modifiedtripleset>
        <mtriple>Aarhus_Airport | cityServed | "Aarhus, Denmark"</mtriple>
      </modifiedtripleset>
      <lex comment="good" lid="Id1">The Aarhus is the airport of Aarhus, Denmark.</lex>
      <lex comment="good" lid="Id2">Aarhus Airport serves the city of Aarhus, Denmark.</lex>
    </entry>
    '''
    entry = ET.Element('entry',attrib={
        'category':'Conceptnet',
        'eid':str(index),
        'size':"1"
    })
    tripleset_text = start+" | "+relationship+" | "+end
    # Original tripleset
    original_tripleset = ET.SubElement(entry,'originaltripleset')
    otriple = ET.SubElement(original_tripleset,'otriple')
    otriple.text = tripleset_text
    # Modified tripleset
    modified_tripleset = ET.SubElement(entry, 'modifiedtripleset')
    mtriple = ET.SubElement(modified_tripleset,'mtriple')
    mtriple.text = tripleset_text
    # Lex entries
    q = CNQuery()
    res = q.query(start,end,relationship)
    for idx, result in enumerate(res['edges']):
        text = result['surfaceText']
        if text is None:
            continue
        #TODO clean the text up.
        lex = ET.SubElement(entry,'lex',attrib={
        'comment':'good',
        'lid':str(idx),
        })
        text = text.replace("[","").replace("]","")
        lex.text = text
    found_lex = False
    for element in entry.getchildren():
        if "lex" in element.tag:
            found_lex = True
            break
    if not found_lex:
        return None
    return entry

def create_conceptnet_xml():

    # create the file structure
    data = ET.Element('benchmark')
    items = ET.SubElement(data, 'entries')
    i = 0
    j=0
    with open(filename) as f:
        for line in f:
            split = line.split("\t")
            rel = split[1]
            end = split[2]
            start = split[3]
            # print(split)
            if "/c/en/" in start and "/c/en/" in end:
                conversion = xml_entry(start.replace("/c/en/",""),end.replace("/c/en/",""),rel,i)
                if conversion is None: continue
                else:
                    items.append(conversion)
                    i+=1
            j+=1
            if j%10000==0:
                print(j)
                mydata = ET.tostring(data)
                myfile = open("items2.xml", "wb")
                myfile.write(mydata)

    mydata = ET.tostring(data)
    myfile = open("items2.xml", "w")
    myfile.write(mydata)

def parse_fun(dict_res):
    items = None
    # with open(dir_write+conceptname.replace("/","__")+".xml","wb") as myfile:
    data = ET.Element('benchmark')
    items = ET.SubElement(data, 'entries')

    i = 1
    #Check connections
    for edge in dict_res["edges"]:
        start = edge["start"]["term"]
        end = edge["end"]["term"]
        rel = edge["rel"]["@id"]
        text = edge["surfaceText"]
        if not text == "":
            if text is None:
                continue
        #Build xml
        entry = ET.Element('entry', attrib={
            'category': 'Conceptnet',
            'eid': str(id),
            'size': "1"
        })
        tripleset_text = start + " | " + rel + " | " + end
        # Original tripleset
        original_tripleset = ET.SubElement(entry, 'originaltripleset')
        otriple = ET.SubElement(original_tripleset, 'otriple')
        otriple.text = tripleset_text
        # Modified tripleset
        modified_tripleset = ET.SubElement(entry, 'modifiedtripleset')
        mtriple = ET.SubElement(modified_tripleset, 'mtriple')
        mtriple.text = tripleset_text


        # TODO clean the text up.
        lex = ET.SubElement(entry, 'lex', attrib={
            'comment': 'good',
            'lid': str(i),
        })
        text = text.replace("[", "").replace("]", "")
        lex.text = text
        items.append(entry)
        i+=1
        # myfile.write(ET.tostring(data))
    return items

def concept_xml(tup):
    conceptname, id = tup
    dir_write = "cn_rdf/"

    query = CNQuery().query_custom_parse(conceptname,None,None,parse_fun)
    # print("Done",id,conceptname)
    return query


def multithread_build(vocabulary_loc, thread_amount=8):
    p = Pool(14)
    concept_list = []
    vocab = pd.read_hdf(vocabulary_loc)
    max = 100
    i = 0
    print("Loading vocab")
    for line in vocab.index:
        # if i == max:
        #     break
        concept_list.append(line)
        i+=1
    print("Mapping")
    results = p.map(concept_xml,zip(concept_list,range(len(concept_list))))
    print(results)

    with open("cn_rdf/cn.xml","wb") as cnfile:
        data = ET.Element('benchmark')
        items = ET.SubElement(data, 'entries')

        for item in results:
            if len(item.getchildren())==0:
                continue
            else:
                for child in item.getchildren():
                    items.append(child)
        cnfile.write(ET.tostring(data))
if __name__ == '__main__':
    # create_conceptnet_xml()
    vocabulary_loc = "fasttext_model/attract_repel.hd5clean"
    multithread_build(vocabulary_loc,4)
