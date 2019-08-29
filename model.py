import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q jieba-fast')
os.system('pip3 install -q optuna<=0.14')

import re

import jieba_fast as jieba
import lightgbm as lgb
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

FULL_WIDTH_SYMBOLS = ''.join(chr(0xff01 + i) for i in range(94))
HALF_WIDTH_SYMBOLS = ''.join(chr(0x21 + i) for i in range(94))
F2H = str.maketrans(FULL_WIDTH_SYMBOLS, HALF_WIDTH_SYMBOLS)

# EMAIL_ADDRESS_PATTERN = re.compile(r'[\w.+-]+@[\w-]+(\.[\w-]+)+')
# URL_PATTERN = re.compile(r'http(s)?[^\s]+[\w]')
# NUMBER_PATTERN = re.compile(r'[+-]?\d+(\.\d+)?')
SYMBOL_PATTERN = re.compile(r'[!-/:-@[-`{-~]')
OTHER_PATTERN = re.compile(r'[^\w\s]')

CHINESE_STOP_WORDS = frozenset([
    "、","。","〈","〉","《","》","一","一些","一何","一切","一则","一方面","一旦","一来","一样","一般","一转眼","七","万一","三","上","上下","下","不","不仅","不但","不光","不单","不只","不外乎","不如","不妨","不尽","不尽然","不得","不怕","不惟","不成","不拘","不料","不是","不比","不然","不特","不独","不管","不至于","不若","不论","不过","不问","与","与其","与其说","与否","与此同时","且","且不说","且说","两者","个","个别","中","临","为","为了","为什么","为何","为止","为此","为着","乃","乃至","乃至于","么","之","之一","之所以","之类","乌乎","乎","乘","九","也","也好","也罢","了","二","二来","于","于是","于是乎","云云","云尔","五","些","亦","人","人们","人家","什","什么","什么样","今","介于","仍","仍旧","从","从此","从而","他","他人","他们","他们们","以","以上","以为","以便","以免","以及","以故","以期","以来","以至","以至于","以致","们","任","任何","任凭","会","似的","但","但凡","但是","何","何以","何况","何处","何时","余外","作为","你","你们","使","使得","例如","依","依据","依照","便于","俺","俺们","倘","倘使","倘或","倘然","倘若","借","借傥然","假使","假如","假若","做","像","儿","先不先","光是","全体","全部","八","六","兮","共","关于","关于具体地说","其","其一","其中","其二","其他","其余","其它","其次","具体地说","具体说来","兼之","内","再","再其次","再则","再有","再者","再者说","再说","冒","冲","况且","几","几时","凡","凡是","凭","凭借","出于","出来","分","分别","则","则甚","别","别人","别处","别是","别的","别管","别说","到","前后","前此","前者","加之","加以","即","即令","即使","即便","即如","即或","即若","却","去","又","又及","及","及其","及至","反之","反而","反过来","反过来说","受到","另","另一方面","另外","另悉","只","只当","只怕","只是","只有","只消","只要","只限","叫","叮咚","可","可以","可是","可见","各","各个","各位","各种","各自","同","同时","后","后者","向","向使","向着","吓","吗","否则","吧","吧哒","含","吱","呀","呃","呕","呗","呜","呜呼","呢","呵","呵呵","呸","呼哧","咋","和","咚","咦","咧","咱","咱们","咳","哇","哈","哈哈","哉","哎","哎呀","哎哟","哗","哟","哦","哩","哪","哪个","哪些","哪儿","哪天","哪年","哪怕","哪样","哪边","哪里","哼","哼唷","唉","唯有","啊","啐","啥","啦","啪达","啷当","喂","喏","喔唷","喽","嗡","嗡嗡","嗬","嗯","嗳","嘎","嘎登","嘘","嘛","嘻","嘿","嘿嘿","四","因","因为","因了","因此","因着","因而","固然","在","在下","在于","地","基于","处在","多","多么","多少","大","大家","她","她们","好","如","如上","如上所述","如下","如何","如其","如同","如是","如果","如此","如若","始而","孰料","孰知","宁","宁可","宁愿","宁肯","它","它们","对","对于","对待","对方","对比","将","小","尔","尔后","尔尔","尚且","就","就是","就是了","就是说","就算","就要","尽","尽管","尽管如此","岂但","己","已","已矣","巴","巴巴","年","并","并且","庶乎","庶几","开外","开始","归","归齐","当","当地","当然","当着","彼","彼时","彼此","往","待","很","得","得了","怎","怎么","怎么办","怎么样","怎奈","怎样","总之","总的来看","总的来说","总的说来","总而言之","恰恰相反","您","惟其","慢说","我","我们","或","或则","或是","或曰","或者","截至","所","所以","所在","所幸","所有","才","才能","打","打从","把","抑或","拿","按","按照","换句话说","换言之","据","据此","接着","故","故此","故而","旁人","无","无宁","无论","既","既往","既是","既然","日","时","时候","是","是以","是的","更","曾","替","替代","最","月","有","有些","有关","有及","有时","有的","望","朝","朝着","本","本人","本地","本着","本身","来","来着","来自","来说","极了","果然","果真","某","某个","某些","某某","根据","欤","正值","正如","正巧","正是","此","此地","此处","此外","此时","此次","此间","毋宁","每","每当","比","比及","比如","比方","没奈何","沿","沿着","漫说","焉","然则","然后","然而","照","照着","犹且","犹自","甚且","甚么","甚或","甚而","甚至","甚至于","用","用来","由","由于","由是","由此","由此可见","的","的确","的话","直到","相对而言","省得","看","眨眼","着","着呢","矣","矣乎","矣哉","离","秒","竟而","第","等","等到","等等","简言之","管","类如","紧接着","纵","纵令","纵使","纵然","经","经过","结果","给","继之","继后","继而","综上所述","罢了","者","而","而且","而况","而后","而外","而已","而是","而言","能","能否","腾","自","自个儿","自从","自各儿","自后","自家","自己","自打","自身","至","至于","至今","至若","致","般的","若","若夫","若是","若果","若非","莫不然","莫如","莫若","虽","虽则","虽然","虽说","被","要","要不","要不是","要不然","要么","要是","譬喻","譬如","让","许多","论","设使","设或","设若","诚如","诚然","该","说","说来","请","诸","诸位","诸如","谁","谁人","谁料","谁知","贼死","赖以","赶","起","起见","趁","趁着","越是","距","跟","较","较之","边","过","还","还是","还有","还要","这","这一来","这个","这么","这么些","这么样","这么点儿","这些","这会儿","这儿","这就是说","这时","这样","这次","这般","这边","这里","进而","连","连同","逐步","通过","遵循","遵照","那","那个","那么","那么些","那么样","那些","那会儿","那儿","那时","那样","那般","那边","那里","都","鄙人","鉴于","针对","阿","除","除了","除外","除开","除此之外","除非","随","随后","随时","随着","难道说","零","非","非但","非徒","非特","非独","靠","顺","顺着","首先"
])


class BasePreprocessor(object):
    pass


class EnglishPreprocessor(BasePreprocessor):
    def __call__(self, doc):
        doc = doc.lower()
        # doc = EMAIL_ADDRESS_PATTERN.sub('EMAILADDRESS', doc)
        # doc = URL_PATTERN.sub('URL', doc)
        # doc = NUMBER_PATTERN.sub('NUMBER', doc)
        doc = SYMBOL_PATTERN.sub(' ', doc)
        doc = OTHER_PATTERN.sub('', doc)

        return doc.strip()


class ChinesePreprocessor(BasePreprocessor):
    def __call__(self, doc):
        doc = doc.translate(F2H)
        doc = doc.lower()
        # doc = EMAIL_ADDRESS_PATTERN.sub('EMAILADDRESS', doc)
        # doc = URL_PATTERN.sub('URL', doc)
        # doc = NUMBER_PATTERN.sub('NUMBER', doc)
        doc = SYMBOL_PATTERN.sub(' ', doc)

        return doc.strip()


class BaseTokenizer(object):
    pass


class ChineseTokenizer(BaseTokenizer):
    def __init__(self, hmm=True):
        self.hmm = hmm

    def __call__(self, doc):
        return jieba.cut(doc, HMM=self.hmm)


class Model(object):
    def __init__(
        self,
        metadata,
        class_weight='balanced',
        dtype=np.float32,
        early_stopping_rounds=10,
        hmm=False,
        max_df=0.95,
        max_features=100_000,
        max_samples=100_000,
        min_df=2,
        n_components=100,
        n_estimators=1_000,
        n_jobs=-1,
        random_state=0
    ):
        self.class_weight = class_weight
        self.done_training = False
        self.dtype = dtype
        self.early_stopping_rounds = early_stopping_rounds
        self.hmm = hmm
        self.max_df = max_df
        self.max_features = max_features
        self.max_samples = max_samples
        self.metadata = metadata
        self.min_df = min_df
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

    def train(self, train_dataset, remaining_time_budget=None):
        X, y = train_dataset
        y = np.argmax(y, axis=1)

        if self.metadata['train_num'] > self.max_samples:
            X, _, y, _ = train_test_split(
                X,
                y,
                random_state=self.random_state,
                train_size=self.max_samples
            )

        if self.metadata['language'] == 'ZH':
            preprocessor = ChinesePreprocessor()
            stop_words = CHINESE_STOP_WORDS
            tokenizer = ChineseTokenizer(hmm=self.hmm)
        else:
            preprocessor = EnglishPreprocessor()
            stop_words = 'english'
            tokenizer = None

        self.vectorizer_ = TfidfVectorizer(
            dtype=self.dtype,
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            preprocessor=preprocessor,
            stop_words=stop_words,
            tokenizer=tokenizer
        )
        self.reducer_ = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.model_ = lgb.LGBMClassifier(
            class_weight=self.class_weight,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        X = self.vectorizer_.fit_transform(X)
        X = self.reducer_.fit_transform(X)
        X, X_valid, y, y_valid = train_test_split(
            X,
            y,
            random_state=self.random_state
        )

        self.model_.fit(
            X,
            y,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_set=[(X_valid, y_valid)]
        )

        self.done_training = True

    def test(self, X, remaining_time_budget=None):
        X = self.vectorizer_.transform(X)
        X = self.reducer_.transform(X)

        return self.model_.predict_proba(X)
