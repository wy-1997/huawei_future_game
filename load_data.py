import json
import random
import numpy as np
import time
from finetune_config import config

class data_loader():
    def __init__(self, model):
        self.data_dir = "./data/"  
        self.load_data()
        self.oagbert = model

    def get_examples(self, data_dict):
        train_data, valid_data, test_data = {}, {}, {}
        for ins in data_dict:
            pub_id = ins['pub_id']
            experts = ins['experts']
            if len(experts) >= 3:
                train_experts = experts[0:len(experts)-2]
                valid_experts = [experts[-2]]
                test_experts = [experts[-1]]
                if pub_id in train_data:
                    print("repeat train pub_id....")
                else:
                    train_data[pub_id] = train_experts
                if pub_id in valid_data:
                    print("repeat valid pub_id...")
                else:
                    valid_data[pub_id] = valid_experts
                if pub_id in test_data:
                    print("repeat test pub_id....")
                else:
                    test_data[pub_id] = test_experts
            else:
                train_experts = experts
                if pub_id in train_data:
                    print("repeat train pub_id...")
                else:
                    train_data[pub_id] = train_experts
        return train_data, valid_data, test_data 

    
    def get_papers(self, data_dict):
        paper_infos = {}
        for iid in data_dict:
            paper_id, title, abstract, keywords, year = iid["id"], iid["title"], iid["abstract"], iid["keywords"], iid["year"]
            # process keywords
            str_keywords = ""
            if keywords != "":
                for word in keywords:
                    if word == keywords[0]:
                        str_keywords = word
                    else:
                        str_keywords = str_keywords + ' ' + word

            # check data -- remove unexist title and abstract data
            if title == "" and abstract =="" and str_keywords == "":
                print("unexisting title and abstract and keywords paper data....")
            else:
                infos = {"title": title, "abstract": abstract, "keywords": str_keywords, "year": year}
                if paper_id in paper_infos:
                    print("repeat paper id.....")
                else:
                    paper_infos[paper_id] = infos
        return paper_infos


    def get_experts(self, data_dict):
        expert_infos = {}
        for expert in data_dict:
            expert_id = expert['id']
            pub_info = expert["pub_info"]
            if expert.get("interests", None) != None:
                interests = expert["interests"]
            elif expert.get("tags", None) != None:
                tags = expert["tags"]
                interests = tags
            else:
                interests = []
            
            # process interests
            str_interests = ""
            len_interests = len(interests)
            if len_interests != 0:
                for interest in interests:
                    if interest == interests[0]:
                        str_interests = interest['t']
                    else:
                        str_interests = str_interests + ' ' + interest['t']

            # process pub_info
            pub_infos = {}
            for iid in pub_info:
                pid = iid["id"]
                if iid.get('title', None) != None:
                    title = iid['title']
                else:
                    title = ""
                if iid.get("abstract", None) != None:
                    abstract = iid["abstract"]
                else:
                    abstract = ""
                if iid.get("keywords", None) != None:
                    keywords = iid["keywords"]
                else:
                    keywords = []
                infos = {"title": title, "abstract": abstract, "keywords": keywords}
                if pid in pub_infos:
                    pass
                else:
                    if title == "" and abstract == "" and keywords == []:
                        print("pub_info.....", iid)
                        pass
                    else:
                        pub_infos[pid] = infos

            # check expert data remove data unexist interests and pub_info
            if str_interests == "" and pub_infos == {}:
                print("unexisting_id", expert_id)
            else:
                if expert_id in expert_infos:
                    pass
                else:
                    info = {"interests": str_interests, "pub_info": pub_infos}
                    expert_infos[expert_id] = info

        return expert_infos

    def get_data_pairs(self, data_dict):
        data_pairs = []
        for pid, experts in data_dict.items():
            for expert in experts:
                if expert in self.experts_infos:
                    data_pairs.append((pid, expert))
                else:
                    print("expert not exist in experts_infos......")
        return data_pairs


    def load_data(self):
        """
        get paper info
        """
        with open(self.data_dir+'training_set/publications.json', 'r',encoding='utf-8') as f:
            self.raw_paper_infos = json.load(f)
        print("self.raw_paper_infos", len(self.raw_paper_infos))
        self.paper_infos = self.get_papers(self.raw_paper_infos)
        print("self.paper_infos", len(self.paper_infos.keys()))

        """
        get experts info
        """
        with open(self.data_dir+'experts/experts0.json', 'r',encoding='utf-8') as f:
            self.raw_experts_0 = json.load(f)
        print("self.raw_experts_0", len(self.raw_experts_0))    
        self.experts_infos_0 = self.get_experts(self.raw_experts_0) 
        print("self.experts_infos_0", len(list(self.experts_infos_0.keys())))

        with open(self.data_dir+'experts/experts1.json', 'r',encoding='utf-8') as f:
            self.raw_experts_1 = json.load(f)
        print("self.raw_experts_1", len(self.raw_experts_1))    
        self.experts_infos_1 = self.get_experts(self.raw_experts_1) 
        print("self.experts_infos_1", len(list(self.experts_infos_1.keys())))

        with open(self.data_dir+'experts/experts2.json', 'r',encoding='utf-8') as f:
            self.raw_experts_2 = json.load(f)
        print("self.raw_experts_2", len(self.raw_experts_2))    
        self.experts_infos_2 = self.get_experts(self.raw_experts_2) 
        print("self.experts_infos_2", len(list(self.experts_infos_2.keys())))

        with open(self.data_dir+'experts/experts3.json', 'r',encoding='utf-8') as f:
            self.raw_experts_3 = json.load(f)
        print("self.raw_experts_3", len(self.raw_experts_3))    
        self.experts_infos_3 = self.get_experts(self.raw_experts_3) 
        print("self.experts_infos_3", len(list(self.experts_infos_3.keys())))


        with open(self.data_dir+'experts/experts4.json', 'r',encoding='utf-8') as f:
            self.raw_experts_4 = json.load(f)
        print("self.raw_experts_4", len(self.raw_experts_4))    
        self.experts_infos_4 = self.get_experts(self.raw_experts_4) 
        print("self.experts_infos_4", len(list(self.experts_infos_4.keys())))

        self.experts_infos = {**self.experts_infos_0, **self.experts_infos_1, **self.experts_infos_2, **self.experts_infos_3, **self.experts_infos_4}
        print("self.experts_infos", len(list(self.experts_infos.keys())))

        """
        check experts_infos
        """
        for e_id, e_infos in self.experts_infos.items():
            interests = e_infos["interests"]
            pub_infos = e_infos["pub_info"]
            if interests == "" and pub_infos == {}:
                print("e_id##########", e_id)
                print("e_infos.....", e_infos)
        """
        get train, valid, test data form training_set
        """
        with open(self.data_dir+'training_set/results.json', 'r',encoding='utf-8') as f:
            self.raw_data_dict = json.load(f)
        # train_data, valid_data, test_data
        self.train_data_dict, self.valid_data_dict, self.test_data_dict = self.get_examples(self.raw_data_dict) 
        print("Train:{} Valid:{} Test:{}".format(len(self.train_data_dict.keys()), len(self.valid_data_dict.keys()), len(self.test_data_dict.keys())))
        self.train_data = self.get_data_pairs(self.train_data_dict) 
        self.valid_data = self.get_data_pairs(self.valid_data_dict) 
        self.test_data = self.get_data_pairs(self.test_data_dict) 



    def get_batch(self, all_data, batch_size):
        # paper - experts_list
        random.shuffle(all_data)
        batch_data = []
        data_len = len(all_data)
        num_batch = int(data_len // batch_size)
        for i in range(0, num_batch):
            batch_data.append(all_data[i*batch_size:(i+1)*batch_size])
        if num_batch * batch_size < data_len:
            add_len = (num_batch+1) * batch_size - data_len
            add_data = all_data[num_batch*batch_size:data_len]
            add_data.extend(all_data[0:add_len])
            batch_data.append(add_data)
        return batch_data

    def generate_negs_data(self, paper_infos, experts_infos, neg_num):
        batch_infos_neg = []
        # batch share negs
        # random.sample K negs 
        negs_id = random.sample(list(experts_infos.keys()), neg_num)
        negs_infos = {}
        for neg in negs_id:
            if neg in negs_infos:
                print("repeat negs in negs_info....")
            else:
                negs_infos[neg] = experts_infos[neg]
                if experts_infos[neg]["interests"] == "" and experts_infos[neg]["pub_info"] == {}:
                    print("experts_empty", experts_infos[neg])
        batch_infos_neg.append(negs_infos)

        return batch_infos_neg


    def generate_batch_data_test(self, paper_infos, experts_infos, batch, neg_num):
        batch_infos_anchor = []
        batch_infos_pos = []
        batch_infos_neg = []
        # generate anchor, pos, neg 
        batch_experts = []
        for pid, eid in batch:
            p_infos = paper_infos[pid]
            
            e_infos = experts_infos[eid]

            # build anchor, pos, neg
            # anchor
            anchor_infos = {}
            if pid in anchor_infos:
                print("repeat pid....")
            else:
                anchor_infos[pid] = p_infos
            batch_infos_anchor.append(anchor_infos)

            # pos
            pos_infos = {}
            if pid in pos_infos:
                print("repeat pid in pos_infos...")
            else:
                pos_infos[pid] = e_infos
            batch_infos_pos.append(pos_infos)
            batch_experts.extend(self.train_data_dict[pid])

        # batch share negs
        # random.sample K negs 
        #print("batch_experts", batch_experts)
        negs_id = random.sample(list(set(experts_infos.keys()) - set(batch_experts)), neg_num)
        #print("negs_id", negs_id)
        negs_infos = {}
        for neg in negs_id:
            if neg in negs_infos:
                print("repeat negs in negs_info....")
            else:
                negs_infos[neg] = experts_infos[neg]
                if experts_infos[neg]["interests"] == "" and experts_infos[neg]["pub_info"] == {}:
                    print("experts_empty", experts_infos[neg])
        batch_infos_neg.append(negs_infos)

        return batch_infos_anchor, batch_infos_pos, batch_infos_neg


    def generate_batch_data(self, paper_infos, experts_infos, batch, neg_num):
        batch_infos_anchor = []
        batch_infos_pos = []
        batch_infos_neg = []
        # generate anchor, pos, neg
        for pid, eid in batch:
            p_infos = paper_infos[pid]
            e_infos = experts_infos[eid]

            # build anchor, pos, neg
            # anchor
            anchor_infos = {}
            if pid in anchor_infos:
                print("repeat pid....")
            else:
                anchor_infos[pid] = p_infos
            batch_infos_anchor.append(anchor_infos)

            # pos
            pos_infos = {}
            if pid in pos_infos:
                print("repeat pid in pos_infos...")
            else:
                pos_infos[pid] = e_infos
            batch_infos_pos.append(pos_infos)

            # negs
            # random.sample K negs 
            negs_id = random.sample(list(set(experts_infos.keys()) - set(self.train_data_dict[pid])), neg_num)
            negs_infos = {}
            for neg in negs_id:
                if neg in negs_infos:
                    print("repeat negs in negs_info....")
                else:
                    negs_infos[neg] = experts_infos[neg]
                    if experts_infos[neg]["interests"] == "" and experts_infos[neg]["pub_info"] == {}:
                        print("experts_empty", experts_infos[neg])
                    """
                    while True:
                        if experts_infos[neg]["interests"] != "":
                            negs_infos[neg] = experts_infos[neg]
                            break
                        # check pub_info when interests == ""
                        elif experts_infos[neg]["pub_info"] != {}:
                            negs_infos[neg] = experts_infos[neg]
                            break
                        # unexisting interests and pub_info (the information for this expert is useless)
                        else:
                            # sample another neg 
                            print("experts_infos[neg]", experts_infos[neg])
                            print("expert_id", neg)
                            print("candidate_negs", len(list(set(experts_infos.keys()) - set(self.train_data_dict[pid]) - set(negs_id))))
                            neg = random.sample(list(set(experts_infos.keys()) - set(self.train_data_dict[pid]) - set(negs_id)), 1)[0]
                            print("neg_another", neg)
                    """
            batch_infos_neg.append(negs_infos)

        return batch_infos_anchor, batch_infos_pos, batch_infos_neg



    def get_batch_tokens(self, infos, flag):
        batch_tokens = []
        for info in infos:
            tokens_dict = {}
            for p_id, p_info in info.items():
                # get tokens
                tokens = self.build_bert_inputs(p_info, flag)
                #print("tokens...", tokens)
                if p_id in tokens_dict:
                    print("repeat p_id.....")
                else:
                    tokens_dict[p_id] = tokens
            batch_tokens.append(tokens_dict)
        return batch_tokens


    def build_bert_inputs(self, p_info, flag):
        if flag == "anchor":
            # title & abstract & keywords
            if p_info.get("abstract", None) != None:
                abstract = p_info["abstract"]
            else:
                abstract = ""
            if p_info.get("title", None) != None:
                title = p_info["title"]
            else:
                title = ""
            if p_info.get("keywords", None) != None:
                keywords = p_info["keywords"]
            else:
                keywords = ""
            return self.oagbert.build_inputs(title=title, abstract=abstract, concepts=keywords)
        elif flag == "pos" or flag == "neg":
            # experts
            if p_info.get("interests", None) != None:
                interests = p_info["interests"]
            else:
                interests = ""
            e_tokens = self.oagbert.build_inputs(title=interests, abstract="")
            # prcoess experts'pub infos
            expert_pub_infos = p_info["pub_info"]
            pub_tokens = {}
            for pid, expert_pub_info in expert_pub_infos.items():
                title = expert_pub_info['title']
                abstract = expert_pub_info['abstract']
                keywords_list = expert_pub_info['keywords']
                keywords = ""
                if len(keywords_list) != 0:
                    for word in keywords_list:
                        if word == keywords_list[0]:
                            keywords = word
                        else:
                            keywords = keywords + ' ' + word
                p_tokens = self.oagbert.build_inputs(title=title, abstract=abstract, concepts=keywords)
                if pid in pub_tokens:
                    print("repeat pid....")
                else:
                    pub_tokens[pid] = p_tokens
            tokens = {"interests": e_tokens, "pub_info": pub_tokens}
            return tokens 
        else:
            raise Exception("undefine flag")
            return 

     


