# ML_IDEA
一些机器学习和深度学习的经验


1.在做kaggle比赛时，要注意使用kaggle wiki，kaggle wiki列举了比赛使用的一些评判标准以及历史上使用该评判标准的比赛供我们参考。（tips：将评判标准优化为loss function）

2.工业上一些用机器学习的例子：1) 互联网用户行为：CTR预测；2) 销量预测：比如阿里菜鸟对存货数量的预测；3) 图像内容在比赛中一般用DL垄断；4) 推荐系统相关：电商推荐 等。

3.强化学习主要在机器人领域。

4.一些方法讲解：1) Naive_Bayes:当数据集处理的很好时，且数据分布均匀时，这个方法虽然简单但处理NLP能有很好的效果；2) SVM：在DL出现前垄断classification，在小数据集上表现好；3) SVD,PCA,K-means：工业上多用这些方法进行辅助，比如先用K-means对用户做聚类，将cluster_id作为新的特征加入到模型中；4) 关联规则方法有Apriori,FP-Growth; 5) 隐马在RNN之后用的很少；6) 数据太多的情况下使用SGD，需要根据数据量级去选择方法。

5.实战常用工具：1) sklearn: 全，但是速度不一定最快；2）gensim: NLP；3）XGBoost: 主要是boosting; 4) pandas: 数据清洗，产出特征； 5）Natural Language Toolkit: 英文NLP（中文用的不多，因为中文要做分词);  6）Tensorflow: 占内存多，速度不是很快； 7）Caffe: 主要是图像处理；8）MxNet: 和 XGBoost一家； 9）Keras: DL, 接口简单。

6.在处理问题这个过程中，处理数据和特征占据70%(包括认识数据(当数据维度很高，难以可视化时，可以使用t-SNE这个高维可视化方式，或者使用PCA等方法降维)、数据预处理(清洗、调权，这包括去除离群点和bad点，看数据分布是否均匀等)、特征工程)，而模型建立占据30%。

7.数据预处理：1）数据清洗: 不可信样本丢掉(根据实际问题）、缺失值极多字段考虑不用；2) 数据采样：上采样/下采样(是为了保证样本均衡，比如正负样本比例为1：10，而评判标准为正确率，那么我们将所有样本判为负的，正确率就有90%，这样的分类器没有意义)、保证样本均衡(样本比例1：2，2：3都ok，不像1：10甚至再往上那种程度就可以)。

保证样本均衡的方法如下：1) 上采样：加大正样本权重(把正样本重复若干次)；2）loss 函数中加大正样本的权重；3）将负样本平均分为10份，和正样本结合做10个分类器，最后再对这10个分类器做一个ensemble。

8.载入数据工具：若数据量不是那么大，可以载入内存，那么使用pandas(数据量很大时，不要批量操作特征，可以抽取单个特征做转换)；数据量很大时，可以使用hive sql, spark sql。

9.特征工程: 1）填补缺失值：对于缺失值很少的，一般可以使用mode或mean去填补；对于缺失值不是那么多的，或者缺失值代表某种意义(代表没有），可以将缺失值作为一个特征，比如颜色这个特征有三种颜色，用onehot编码时，可以编成4位，最后一位表示是否有颜色，0表示有，而1表示无。

                2）数据变换：可以采取log、指数、box-cox对数据做变换，使变换完的数据分布更加明显。

                3）特征处理：对于不同的特征有不同的处理方法，比如时间类，根据具体问题，可以有间隔型(每个日子离双11还有几天)、组合型(每周点击率）、离散化(将每天时间段分为饭点和非饭点，可以将饭点标为1，非饭点标为0）；对于文本型，可以使用n-gram, bag of words, TF-IDF(编码体现每个词是否出现以及其影响力)。

                4）特征抽取：sklearn.feature_extraction(文本型)；scaling表示特征缩放，并不要求每个特征范围一致，他的区间可以是[0,3]，不像normalization那么严格在[0,1]上；对于连续变量离散化可以用binarization(二段切分)或pandas.cut作多段切分。

                5）特征选择：点击打开链接 ①过滤型：sklearn.feature_selection.SelectKBest (少用，评估相似度，一般用于LR，实际情况中多用方差，因为方差大直观有作用，方差为0表示数值都相同，那么这个特征可能没那么大作用)；②包裹型：sklearn.feature_selection.RFE ( RFE为Recursive feature elimination缩写，将特征按重要程度排序剔除末尾一部分变量)；③嵌入型: sklearn.feature_selection.SelectFromModel (有些模型提供coef或者feature_importance_供选择特征，比如tree或者xgboost )，L1正则化(多用于LinearSVC & LogitR)。

10.模型选择：用CV选择模型，用GS在模型中调参。可使用sklearn.grid_search.GridSearchCV，但GS和CV一起用时效果很慢，K_fold可以优化用来做特征选择。

11.学习状态评估：通过画learning_curve评判模型是过拟合还是欠拟合，根据不同状态对模型进行修正。plot learning curve

12.模型融合：model ensemble stacking：用多种predictor结果作为特征训练(神经网络就是这样)；若使用GBDT不要使用sklearn,很慢，建议使用XGBoost或者LightGBM。

13.XGBoost库使用：基本应用：xgboost/demo/guide-python/basic_walkthrough.py （包括可载入数据类型及给数据加权）；高级应用：自己设定objective函数(xgboost/demo/guide-python/custom_objective.py)，注意在定义函数时，函数要求可导，且要告诉xgboost该函数的一阶导(grad)和二阶导(hess)。除了XGBoost，还有百度开源的一个基于RF算法的库(在大数据集上使用)，效果比XGBoost好一些，还有Google的LightGBM。
