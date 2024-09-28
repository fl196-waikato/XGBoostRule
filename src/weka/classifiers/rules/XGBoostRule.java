package weka.classifiers.rules;

import weka.classifiers.RandomizableClassifier;
import weka.core.*;

import java.util.*;

/**
 * XGBoostRule classifier: builds a single rule using boosting techniques.
 */
public class XGBoostRule extends RandomizableClassifier implements WeightedInstancesHandler, AdditionalMeasureProducer {
    public Enumeration<String> enumerateMeasures() {
        String[] measures = {"measureNumRules"};
        return Collections.enumeration(Arrays.asList(measures));
    }

    @Override
    public double getMeasure(String measureName) {
        if (measureName.equals("measureNumRules")) {
            return 1; // 返回XGBoostRule 的规则数量,但是这里应该总是1
        }
        throw new IllegalArgumentException("Measure " + measureName + " not supported.");
    }

    private double eta = 0.3;
    private double lambda = 1.0;
    private double gamma = 1.0;
    private double subsample = 0.5;
    private double colsample_bynode = 1.0;
    private int max_length = 6;//应用max_length代表规则长度,其内可包含的最多条件数量, 限制规则的复杂度, 避免过拟合
    private double min_child_weight = 1.0;


    @OptionMetadata(displayName = "eta", description = "Learning rate",
            commandLineParamName = "eta", commandLineParamSynopsis = "-eta <double>", displayOrder = 1)
    public void setEta(double e) {
        eta = e;
    }

    public double getEta() {
        return eta;
    }

    @OptionMetadata(displayName = "lambda", description = "Regularization parameter",
            commandLineParamName = "lambda", commandLineParamSynopsis = "-lambda <double>", displayOrder = 2)
    public void setLambda(double l) {
        lambda = l;
    }

    public double getLambda() {
        return lambda;
    }

    @OptionMetadata(displayName = "gamma", description = "Regularization parameter for rule length",
            commandLineParamName = "gamma", commandLineParamSynopsis = "-gamma <double>", displayOrder = 3)
    public void setGamma(double g) {
        gamma = g;
    }

    public double getGamma() {
        return gamma;
    }

    @OptionMetadata(displayName = "subsample", description = "Subsample ratio of the training instances",
            commandLineParamName = "subsample", commandLineParamSynopsis = "-subsample <double>", displayOrder = 4)
    public void setSubsample(double s) {
        subsample = s;
    }

    public double getSubsample() {
        return subsample;
    }

    @OptionMetadata(displayName = "colsample_bynode", description = "Subsample ratio of columns",
            commandLineParamName = "colsample_bynode", commandLineParamSynopsis = "-colsample_bynode <double>", displayOrder = 5)
    public void setColSampleByNode(double c) {
        colsample_bynode = c;
    }

    public double getColSampleByNode() {
        return colsample_bynode;
    }

    @OptionMetadata(displayName = "max_length", description = "Maximum rule length",
            commandLineParamName = "max_length", commandLineParamSynopsis = "-max_length <int>", displayOrder = 6)
    public void setMaxLength(int m) {
        max_length = m;
    }

    public int getMaxLength() {
        return max_length;
    }

    @OptionMetadata(displayName = "min_child_weight", description = "Minimum sum of instance weight (Hessian) needed in a child",
            commandLineParamName = "min_child_weight", commandLineParamSynopsis = "-min_child_weight <double>", displayOrder = 7)
    public void setMinChildWeight(double w) {
        min_child_weight = w;
    }

    public double getMinChildWeight() {
        return min_child_weight;
    }


    private class SplitSpecification {
        private final Attribute attribute;
        private double splitPoint;
        private double splitQuality;

        private SplitSpecification(Attribute attribute, double splitQuality, double splitPoint) {
            this.attribute = attribute;
            this.splitQuality = splitQuality;
            this.splitPoint = splitPoint;
        }
    }

    private class SufficientStatistics {
        private double sumOfNegativeGradients = 0.0;
        private double sumOfHessians = 0.0;

        private SufficientStatistics(double sumOfNegativeGradients, double sumOfHessians) {
            this.sumOfNegativeGradients = sumOfNegativeGradients;
            this.sumOfHessians = sumOfHessians;
        }

        private void updateStats(double negativeGradient, double hessian, boolean add) {
            sumOfNegativeGradients = (add) ? sumOfNegativeGradients + negativeGradient : sumOfNegativeGradients - negativeGradient;
            sumOfHessians = (add) ? sumOfHessians + hessian : sumOfHessians - hessian;
        }
    }

    private List<Attribute> ruleAttributes = new ArrayList<>();//应用list 存储规则内应用的判断条件变量
    private Map<Attribute, Double> attributeSplitPoints = new HashMap<>();//用hashmap存储规则属性的分裂点
    private double rulePrediction;//代表规则的预测值, 也就是其所属分类或者树的结果预测值

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // Step 1: Sample the data based on subsample and colsample_bynode parameters
        Instances sampledData = sampleData(data, subsample, colsample_bynode);

        // Step 2: Grow the rule based on the sampled data
        growRule(sampledData);
    }

    private Instances sampleData(Instances data, double subsample, double colsample) throws Exception {
        // 实现样本和特征的随机采样逻辑
        // 检查数据集的能力
        getCapabilities().testWithFail(data);

        // 创建一个随机数生成器
        Random rand = new Random(getSeed());

        // 1. 样本采样
        Instances sampledData = new Instances(data);
        if (subsample <= 1.0) {
            sampledData.randomize(rand); // 随机打乱实例顺序
            int sampleSize = (int) (subsample * data.numInstances());
            sampledData = new Instances(sampledData, 0, sampleSize); // 只保留一部分实例
        }

        // 2. 特征采样
        if (colsample <= 1.0) {
            ArrayList<Attribute> selectedAttributes = new ArrayList<>();
            int numAttributesToSelect = (int) (colsample * sampledData.numAttributes());

            // 将所有的属性按随机顺序排列
            List<Integer> attributeIndices = new ArrayList<>();
            for (int i = 0; i < sampledData.numAttributes(); i++) {
                attributeIndices.add(i);
            }
            Collections.shuffle(attributeIndices, rand);

            // 选取前 numAttributesToSelect 个属性
            for (int i = 0; i < numAttributesToSelect; i++) {
                selectedAttributes.add(sampledData.attribute(attributeIndices.get(i)));
            }

            // 创建新的数据集，只包含选定的属性
            Instances filteredData = new Instances(sampledData, sampledData.numInstances());
            for (int i = 0; i < sampledData.numInstances(); i++) {
                double[] values = new double[selectedAttributes.size()];
                Instance instance = sampledData.instance(i);
                for (int j = 0; j < selectedAttributes.size(); j++) {
                    values[j] = instance.value(selectedAttributes.get(j));
                }
                filteredData.add(new DenseInstance(1.0, values));
            }

            // 设置新数据集的属性信息
            filteredData.setClassIndex(sampledData.classIndex());
            sampledData = filteredData;
        }

        return sampledData;
    }



    /**
     * build the rule using the greedy algorithm to optimize the XGBoost objective function.
     */
    public void growRule(Instances data) {

        // 实现规则生成逻辑
        for (int i = 0; i < max_length; i++) {
            SplitSpecification bestSplit = findBestSplit(data,);
            if (bestSplit == null || bestSplit.splitQuality < gamma) {
                break;
            }

            ruleAttributes.add(bestSplit.attribute);
            attributeSplitPoints.put(bestSplit.attribute, bestSplit.splitPoint);

            data = splitData(data, bestSplit);

            if (data.numInstances() * bestSplit.splitQuality < min_child_weight) {
                break;
            }
        }

        rulePrediction = calculateRulePrediction(data);
    }

    private Instances splitData(Instances data, SplitSpecification split) {
        Instances newData = new Instances(data, data.numInstances());
        for (Instance instance : data) {
            if (instance.value(split.attribute) < split.splitPoint) {
                newData.add(instance);
            }
        }
        return newData;
    }

    private double calculateRulePrediction(Instances data) {
        double sumGradient = 0;
        double sumHessian = 0;
        for (Instance instance : data) {
            sumGradient += instance.classValue();
            sumHessian += instance.weight();
        }
        return sumGradient / (sumHessian + lambda);
    }

    /**
     * Classifies an instance using the generated rule.
     */
    public double classifyInstance(Instance instance) {
        for (Attribute attribute : ruleAttributes) {
            if (instance.value(attribute) >= attributeSplitPoints.get(attribute)) {
                return 0.0;
            }
        }
        return rulePrediction;

    }

    /**
     * Finds the best split for a given attribute.
     */
    private SplitSpecification findBestSplit(Instances data, Attribute attribute, SufficientStatistics initialStats) {
        SufficientStatistics statsLeft = new SufficientStatistics(0.0, 0.0);
        SufficientStatistics statsRight = new SufficientStatistics(initialStats.sumOfNegativeGradients, initialStats.sumOfHessians);

        SplitSpecification bestSplit = new SplitSpecification(attribute, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
        double previousValue = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (instance.value(attribute) > previousValue) {
                if (statsLeft.sumOfHessians != 0 && statsRight.sumOfHessians != 0 &&
                        statsLeft.sumOfHessians >= min_child_weight && statsRight.sumOfHessians >= min_child_weight) {
                    double splitQuality = (0.5 * (impurity(statsLeft) + impurity(statsRight) - impurity(initialStats))) - gamma * ruleAttributes.size();
                    if (splitQuality > bestSplit.splitQuality) {
                        bestSplit.splitQuality = splitQuality;
                        bestSplit.splitPoint = (instance.value(attribute) + previousValue) / 2.0;
                    }
                }
                previousValue = instance.value(attribute);
            }
            statsLeft.updateStats(instance.classValue(), instance.weight(), true);
            statsRight.updateStats(instance.classValue(), instance.weight(), false);
        }

        return bestSplit;
    }


    /**
     * Computes the impurity of a split.
     */
    private double impurity(SufficientStatistics stats) {
        return (stats.sumOfHessians <= 0.0) ? 0.0 :
                stats.sumOfNegativeGradients * stats.sumOfNegativeGradients / (stats.sumOfHessians + lambda);
    }


    /**
     * The main method for running this classifier from a command-line interface.
     */
    public static void main(String[] options) {
        runClassifier(new XGBoostRule(), options);
    }
}