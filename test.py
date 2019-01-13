from os.path import join
from mcnn import MCNNModel

mcnn_model = MCNNModel()
src_paper = {'title': 'ArnetMiner: Extraction and Mining of Academic Social Networks.',
             'authors': ['Jie Tang', 'Jing Zhang', 'Limin Yao', 'Juanzi Li', 'Li Zhang', 'Zhong Su']
             }
dsc_papers = [
    {
        'title': 'Arnetminer: extraction and mining of academic social networks',
        'authors': [
            'J Tang', 'J Zhang', 'L Yao', 'J Li', 'L Zhang', 'Z Su'
        ]
    },
    {
        'title': 'Arnetminer: Expertise Oriented Search Using Social Networks.',
        'authors': [
            'Juanzi Li', 'Jie Tang', 'Jing Zhang', 'Qiong Luo', 'Yunhao Liu', 'Mingcai Hong'
        ]
    }
]

fold = 0
model = mcnn_model.create_model_for_multiple_input()
model.load(join(mcnn_model.model_dir, 'cnn_model_{}.mod'.format(fold)))
ypreds = mcnn_model.predict_similarities(src_paper, dsc_papers, model)
print(ypreds)
