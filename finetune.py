import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from cogdl import oagbert
from load_data import *
from finetune_config import config
import random
from modeling import *
from time import time


output_dir = "./oagbert_finetune_chinese_"
batch_size = config["batch_size"]
epochs = config["epochs"]
valid_step = config["valid_step"]



def get_batch_embed(batch_tokens, flag):
    # get embed
    batch_embed = []
    for tokens in batch_tokens:
        for p_id, token in tokens.items():
            if flag == "anchor":
                input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, maksed_positions, num_spans = token
                _, pooled_output = model.bert.forward(
                    input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
                    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).cuda(),
                    attention_mask=torch.LongTensor(input_masks).unsqueeze(0).cuda(),
                    output_all_encoded_layers=False,
                    checkpoint_activations=False,
                    position_ids=torch.LongTensor(position_ids).unsqueeze(0).cuda(),
                    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).cuda())
                batch_embed.append(pooled_output)
            else:
                # experts embedding
                # interests process
                interests = token["interests"]
                #print("interests", interests)
                input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, maksed_positions, num_spans = interests
                if input_ids != []:
                    _, pooled_output_expert_interests = model.bert.forward(
                        input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
                        token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).cuda(),
                        attention_mask=torch.LongTensor(input_masks).unsqueeze(0).cuda(),
                        output_all_encoded_layers=False,
                        checkpoint_activations=False,
                        position_ids=torch.LongTensor(position_ids).unsqueeze(0).cuda(),
                        position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).cuda())
                else:
                    pooled_output_expert_interests = None
                # pub_info of experts process
                pub_info = token["pub_info"]
                pub_info_embed = []
                for pid, p_token in pub_info.items():
                    input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, maksed_positions, num_spans = p_token
                    if input_ids != []:
                        _, pooled_output_expert_pub = model.bert.forward(
                            input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
                            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).cuda(),
                            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).cuda(),
                            output_all_encoded_layers=False,
                            checkpoint_activations=False,
                            position_ids=torch.LongTensor(position_ids).unsqueeze(0).cuda(),
                            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).cuda())
                        pub_info_embed.append(pooled_output_expert_pub)
                
                # merge interests_embed and pub_info_embed
                # here, we use torch.mean() for merge
                # check...
                if pooled_output_expert_interests == None:
                    if len(pub_info_embed) != 0:
                        pooled_output_expert_cat = torch.cat(pub_info_embed)
                    else:
                        pooled_output_expert_cat = None
                else:
                    if len(pub_info_embed) != 0:
                        pooled_output_expert_cat = torch.cat((pooled_output_expert_interests, torch.cat(pub_info_embed)), 0)
                    else:
                        pooled_output_expert_cat = pooled_output_expert_interests
                pooled_output_expert_final = torch.mean(pooled_output_expert_cat, 0).view(1, config['output_dim'])
                batch_embed.append(pooled_output_expert_final)
    return torch.cat(batch_embed, 0)


def evaluate(model, valid_batches, data_loader, paper_infos, experts_infos):
    model.eval()
    mrr = 0.0
    total_count = 0
    with torch.no_grad():
        """
        整个test公用一个negs
        """
        negs = data_loader.generate_negs_data(paper_infos, experts_infos, config["Negs"])
        neg_tokens = data_loader.get_batch_tokens(negs, "neg")
        neg_emb_candidates = get_batch_embed(neg_tokens, "neg")
        """
        share batch negs
        """
        for batch in valid_batches:
            anchor, pos, _ = data_loader.generate_batch_data_test(paper_infos, experts_infos, batch, config["Negs"])
            
            # use too much time
            anchor_tokens = data_loader.get_batch_tokens(anchor, "anchor") 
            pos_tokens = data_loader.get_batch_tokens(pos, "pos")

            # use too much time
            anchor_emb = get_batch_embed(anchor_tokens, "anchor")
            pos_emb = get_batch_embed(pos_tokens, "pos")
            neg_emb = neg_emb_candidates.repeat(len(batch), 1)
        # batch share negs
        #for batch in valid_batches:
        #    anchor, pos, negs = data_loader.generate_batch_data_test(paper_infos, experts_infos, batch, config["Negs"])
            
        #    # use too much time
        #    tt = time()
        #    anchor_tokens = data_loader.get_batch_tokens(anchor, "anchor") 
        #    print("time...", time() - tt)
        #    tt = time()
        #    pos_tokens = data_loader.get_batch_tokens(pos, "pos")
        #    print("time...", time() - tt)
        #    tt = time()
        #    neg_tokens = data_loader.get_batch_tokens(negs, "neg")
        #    print("time...", time() - tt)

        #    # use too much time
        #    tt = time()
        #    anchor_emb = get_batch_embed(anchor_tokens, "anchor")
        #    print("time...", time() - tt)
        #    tt = time()
        #    pos_emb = get_batch_embed(pos_tokens, "pos")
        #    print("time...", time() - tt)
        #    tt = time()
        #    neg_emb_candidates = get_batch_embed(neg_tokens, "neg")
        #    print("time...", time() - tt)
        #    neg_emb = neg_emb_candidates.repeat(len(batch), 1)

            # anchor & pos_embed
            anchor_emb = F.normalize(anchor_emb.view(-1, 1, dim), p=2, dim=2)
            pos_emb = F.normalize(pos_emb.view(-1, 1, dim), p=2, dim=2)
            neg_emb = F.normalize(neg_emb.view(-1, config["Negs"], dim), p=2, dim=2)

            pos_score = torch.bmm(anchor_emb, pos_emb.transpose(1, 2)) # B*1*1
            neg_score = torch.bmm(anchor_emb, neg_emb.transpose(1, 2))  # B*1*Negs

            # logits:B*(1+Negs)
            logits = torch.cat([pos_score, neg_score], dim=2).squeeze()
            logits = logits.cpu().numpy()

            for i in range(batch_size):
                total_count += 1
                logits_single = logits[i]
                rank = np.argsort(-logits_single)
                true_index = np.where(rank==0)[0][0]
                mrr += np.divide(1.0, true_index+1)
        
        mrr /= total_count
       
    return mrr



if __name__ == "__main__":
    # create model
    tokenizer, model = oagbert("./saved/oagbert-v2")
    model = model.cuda()
    projection = MLP(config["output_dim"]).cuda()
    
    
    # load data
    data_loader = data_loader(model)
    # paper-experts
    train_data, valid_data, test_data = data_loader.train_data, data_loader.valid_data, data_loader.test_data
    print("train_data_length", len(train_data))
    print("valid_data_length", len(valid_data))
    print("test_data_length", len(test_data))
    paper_infos = data_loader.paper_infos
    experts_infos = data_loader.experts_infos
    print("experts_infos.length", len(experts_infos.keys()))

    train_batches = data_loader.get_batch(train_data, batch_size)
    valid_batches = data_loader.get_batch(valid_data, 10*batch_size)
    test_batches = data_loader.get_batch(test_data, 10*batch_size)
    print("train_batches", len(train_batches))
    print("valid_batches", len(valid_batches))
    print("test_batches", len(test_batches))
    
    # infoNCE loss
    criterion = infoNCE().cuda()
    optimizer = torch.optim.Adam([{'params':model.parameters()},{'params': projection.parameters()}], lr=config["learning_rate"])

    model.train()
    projection.train()
    best_mrr = -1
    patience = 0

    # finetuning
    for epoch in range(epochs):
        random.shuffle(train_batches)
        batch_loss = []
        batch_num = 0
        for batch in train_batches:
            batch_num += 1
            # get anchor pos neg
            anchor, pos, neg = data_loader.generate_batch_data(paper_infos, experts_infos, batch, config["neg_num"]) 

            anchor_tokens = data_loader.get_batch_tokens(anchor, "anchor") 
            pos_tokens = data_loader.get_batch_tokens(pos, "pos")
            neg_tokens = data_loader.get_batch_tokens(neg, "neg")
            
            anchor_emb = get_batch_embed(anchor_tokens, "anchor")
            pos_emb = get_batch_embed(pos_tokens, "pos")
            neg_emb = get_batch_embed(neg_tokens, "neg")

            # add MLP 
            # infoNCE loss 
            loss = criterion(projection(anchor_emb), projection(pos_emb), projection(neg_emb))
            print("loss...", loss.item())

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_num > 1 and batch_num % valid_step == 0:
                print("evalute.......")
                t_valid = time()
                mrr = evaluate(model, valid_batches, data_loader, paper_infos, experts_infos)
                print("time for valid....", time() - t_valid)
                print("Epoch:{} batch:{} loss:{} mrr:{}".format(epoch, batch_num, loss.item(), mrr))
                if mrr > best_mrr:
                    best_mrr = mrr
                    # save model
                    torch.save(model.state_dict(), output_dir + "oagbert")
                    print("Best Epoch:{} batch:{} loss:{} mrr:{}".format(epoch, batch_num, loss.item(), mrr))
                else:
                    patience += 1
                    if patience > config["patience"]:
                        print("Best Epoch:{} batch:{} loss:{} mrr:{}".format(epoch, batch_num, loss.item(), mrr))

                model.train()
                projection.train()

                
    # test
    model.load_state_dict(torch.load(output_dir + "oagbert"))  
    mrr = perform_valid(model, test_batches, data_loader, paper_infos, experts_infos)
    print("Best mrr: {:.6f}".format(mrr))

            




