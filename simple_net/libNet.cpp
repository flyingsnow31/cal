//
// Created by CuiQinPro on 2023/5/19.
//

#include "libNet.h"

namespace CuiQin {

    // 初始化网络，包括定义矩阵size，权值
    void libNet::initNet(std::vector<int> layer_neuron_num_) {
        layer_neuron_num = layer_neuron_num_;
        // 生成每一层
        layer.resize(layer_neuron_num.size());
        for (int i=0; i<layer.size(); i++) {
            layer[i].create(layer_neuron_num[i], 1, CV_32FC1);
        }
        std::cout << "生成神经元成功！" << std::endl;
        // 生成权重矩阵和偏
        weights.resize(layer.size() - 1);
        bias.resize(layer.size() - 1);
        for (int i=0; i<(layer.size() - 1); ++i) {
            weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1);
            bias[i] = cv::Mat::zeros(layer[i + 1].rows, 1, CV_32FC1);
        }
        std::cout << "初始化权重矩阵和偏置成功" << std::endl;
        std::cout << "初始化网络，完毕" << std::endl;
    }

    // 初始化权重
    void libNet::initWeight(cv::Mat &dst, int type, double a, double b) {
        if (type == 0) { // 单独处理偏置
            randn(dst, a, b);
        } else {
            randu(dst, a, b);
        }
    }

    // 初始化权值，多个
    void libNet::initWeights(int type, double a, double b) {
        for (int i=0; i<weights.size(); ++i) {
            initWeight(weights[i], 0, 0., 0.1);
        }
    }

    // 初始化偏置
    void libNet::initBias(cv::Scalar& bias_) {
        for (int i=0; i<bias.size(); ++i) {
            bias[i] = bias_;
        }
    }

    // 梯度下降下的error值
    void libNet::deltaError() {
        delta_err.resize(layer.size() - 1);
        for (int i = delta_err.size() - 1; i >= 0; i--)
        {
            delta_err[i].create(layer[i + 1].size(), layer[i + 1].type());
            //cv::Mat dx = layer[i+1].mul(1 - layer[i+1]);
            cv::Mat dx = derivativeFunction(layer[i + 1], activation_function);
            //Output layer delta error
            if (i == delta_err.size() - 1)
            {
                delta_err[i] = dx.mul(output_error);
            }
            else  //Hidden layer delta error
            {
                cv::Mat weight = weights[i];
                cv::Mat weight_t = weights[i].t();
                cv::Mat delta_err_1 = delta_err[i];
                delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);
            }
        }
    }

    // 根据loss更新权重
    void libNet::updateWeights() {
        for (int i = 0; i < weights.size(); ++i)
        {
            // 学习率调整更新频率
            cv::Mat delta_weights = learning_rate * (delta_err[i] * layer[i].t());
            cv::Mat delta_bias = learning_rate * delta_err[i];
            weights[i] = weights[i] + delta_weights;
            bias[i] = bias[i] + delta_bias;
        }
    }

    // 激活函数
    cv::Mat libNet::activationFunction(cv::Mat &x, std::string func_type) {
        activation_function = func_type;
        cv::Mat fx;
        if (func_type == "sigmoid")
        {
            fx = sigmoid(x);
        }
        if (func_type == "tanh")
        {
            fx = tanh(x);
        }
        if (func_type == "ReLU")
        {
            fx = ReLU(x);
        }
        return fx;
    }

    // 前向传播过程
    void libNet::forward() {
        for (int i = 0; i < layer_neuron_num.size() - 1; ++i)
        {
            cv::Mat product = weights[i] * layer[i] + bias[i]; // 非线性运算
            layer[i + 1] = activationFunction(product, activation_function); // 通过激活函数决定输入到下一层的值
        }
        calcLoss(layer[layer.size() - 1], target, output_error, loss); // 计算最后一层输出的loss值
    }

    // 反向传播过程
    void libNet::backward() {
        //move this function to the end of the forward().
        //calcLoss(layer[layer.size() - 1], target, output_error, loss);
        deltaError();
        updateWeights();
    }

    /**
     * 训练过程：
     * 1. 接受一个样本（即一个单列矩阵）作为输入，也即神经网络的第一层；
     * 2. 进行前向传播，也即forward()函数做的事情。然后计算loss；
     * 3. 如果loss值小于设定的阈值loss_threshold，则进行反向传播更新阈值；
     * 4. 重复以上过程直到loss小于等于设定的阈值。
     * @param input 输入数据矩阵
     * @param target_ 分类列
     * @param accuracy_threshold 准确率阈值，终止条件
     */
    void libNet::train1(cv::Mat input, cv::Mat target_, float accuracy_threshold)
    {
        if (input.empty())
        {
            std::cout << "数据集为空!" << std::endl;
            return;
        }

        std::cout << "开始训练...停止条件: acc <= " << accuracy_threshold << std::endl;

        cv::Mat sample; // 采样
        if (input.rows == (layer[0].rows) && input.cols == 1) // 只有一列，单个样本
        {
            target = target_; // 标记标签
            sample = input;
            layer[0] = sample; // 只有一列
            forward();
            //backward();
            int num_of_train = 0;
            while (accuracy < accuracy_threshold)
            {
                backward();
                forward();
                num_of_train++;
                if (num_of_train % 500 == 0)
                {
                    std::cout << "Train " << num_of_train << " times" << std::endl;
                    std::cout << "Loss: " << loss << std::endl;
                }
            }
            std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
            std::cout << "Loss: " << loss << std::endl;
            std::cout << "Train sucessfully!" << std::endl;
        }
        else if (input.rows == (layer[0].rows) && input.cols > 1) // 多个样本
        {
            double batch_loss = 0.; // 整体loss
            int epoch = 0;
            while (accuracy < accuracy_threshold)
            {
                batch_loss = 0.;
                for (int i = 0; i < input.cols; ++i)
                {
                    target = target_.col(i);
                    sample = input.col(i);

                    layer[0] = sample;
                    forward();
                    batch_loss += loss;
                    backward();
                }
                // 训练后直接进行测试，计算准确率
                test(input, target_);
                epoch++;
                if (epoch % 10 == 0)
                {
                    std::cout << "Number of epoch: " << epoch << std::endl;
                    std::cout << "Loss sum: " << batch_loss << std::endl;
                }
                // 不断调整学习率可不是一个好方法
                //if (epoch % 100 == 0)
                //{
                //	learning_rate*= 1.01;
                //}
            }
            std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
            std::cout << "Loss sum: " << batch_loss << std::endl;
            std::cout << "Train sucessfully!" << std::endl;
        }
        else // 输入的数据无法矩阵运算
        {
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        }
    }

    // 利用loss停止训练：当loss小于一定值时停止训练，同时绘制曲线图
    void libNet::train2(cv::Mat input, cv::Mat target_, float loss_threshold, int epochs, bool draw_loss_curve)
    {
        if (input.empty())
        {
            std::cout << "数据集为空!" << std::endl;
            return;
        }

        std::cout << "开始训练...停止条件: loss <= " << loss_threshold << std::endl;

        cv::Mat sample;
        if (input.rows == (layer[0].rows) && input.cols == 1)
        {
            target = target_;
            sample = input;
            layer[0] = sample;
            forward();
            //backward();
            int num_of_train = 0;
            while (loss > loss_threshold)
            {
                backward();
                forward();
                num_of_train++;
                if (num_of_train % 500 == 0)
                {
                    std::cout << "Train " << num_of_train << " times" << std::endl;
                    std::cout << "Loss: " << loss << std::endl;
                }
            }
            std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
            std::cout << "Loss: " << loss << std::endl;
            std::cout << "Train sucessfully!" << std::endl;
        }
        else if (input.rows == (layer[0].rows) && input.cols > 1)
        {
            std::cout<<"保存日志位置："<<log_path<<std::endl;
            std::fstream f;
            f.open(log_path, std::ios::out | std::ios::app);
            f<<"epoch,loss_sum"<<std::endl;
            double batch_loss = loss_threshold + 0.01;
            int epoch = 0;
            double minLoss = 100;
            while (batch_loss > loss_threshold && epoch < epochs)
            {
                batch_loss = 0.;
                for (int i = 0; i < input.cols; ++i)
                {
                    target = target_.col(i);
                    sample = input.col(i);
                    layer[0] = sample;

                    forward();
                    backward();

                    batch_loss += loss;
                }

                loss_vec.push_back(batch_loss);

                if (loss_vec.size() >= 2 && draw_loss_curve)
                {
                    draw_curve(board, loss_vec);
                }
                epoch++;
                if (epoch % output_interval == 0)
                {
                    std::cout << "Number of epoch: " << epoch << std::endl;
                    std::cout << "Loss sum: " << batch_loss << std::endl;
                    if (minLoss > batch_loss)
                        minLoss = batch_loss;
                    std::cout << "Loss min: " << minLoss <<std::endl;
                    f<<epoch<<","<<batch_loss<<std::endl;
                }
                if (epoch % 100 == 0)
                {
//                    learning_rate *= fine_tune_factor; // 考虑不用微调
                }
            }
            std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
            std::cout << "Loss sum: " << batch_loss << std::endl;
            std::cout << "Train sucessfully!" << std::endl;
            f.close();
        }
        else
        {
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        }
    }

    /**
     * test()函数的作用就是用一组训练时没用到的样本，对训练得到的模型进行测试，把通过这个模型得到的结果与实际想要的结果进行比较，看正确来说到底是多少，我们希望正确率越多越好。
     * 1. 用一组样本逐个输入神经网络
     * 2. 通过前向传播得到一个输出值
     * 3. 比较实际输出与理想输出，计算正确率
     * @param input
     * @param target_
     */
    void libNet::test(cv::Mat &input, cv::Mat &target_)
    {
        if (input.empty())
        {
            std::cout << "数据集为空!" << std::endl;
            return;
        }
        std::cout << std::endl << "开始测试..." << std::endl;

        if (input.rows == (layer[0].rows) && input.cols == 1)
        {
            int predict_number = predict_one(input);

            cv::Point target_maxLoc;
            minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
            int target_number = target_maxLoc.y;

            std::cout << "Predict: " << predict_number << std::endl;
            std::cout << "Target:  " << target_number << std::endl;
            std::cout << "Loss: " << loss << std::endl;
        }
        else if (input.rows == (layer[0].rows) && input.cols > 1)
        {
            double loss_sum = 0;
            int right_num = 0;
            cv::Mat sample;
            for (int i = 0; i < input.cols; ++i)
            {
                sample = input.col(i);
                int predict_number = predict_one(sample);
                loss_sum += loss;

                target = target_.col(i);
                cv::Point target_maxLoc;
                minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
                int target_number = target_maxLoc.y;

                std::cout << "Test sample: " << i << "   " << "Predict: " << predict_number << "\tTarget:  " << target_number << std::endl << std::endl;
                if (predict_number == target_number)
                {
                    right_num++;
                }
            }
            accuracy = (double)right_num / input.cols;
            std::cout << "Loss sum: " << loss_sum << std::endl;
            std::cout << "accuracy: " << accuracy << std::endl;
        }
        else
        {
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
            return;
        }
    }

    /**
     * 预测，给定一个输入，给出想要的输出值。其中包含了对forward()函数的调用
     * @param input
     * @return
     */
    int libNet::predict_one(cv::Mat &input)
    {
        if (input.empty())
        {
            std::cout << "Input is empty!" << std::endl;
            return -1;
        }

        if (input.rows == (layer[0].rows) && input.cols == 1)
        {
            layer[0] = input;
            forward();

            cv::Mat layer_out = layer[layer.size() - 1];
            cv::Point predict_maxLoc;

            //前向传播得到最后一层输出层layer_out，然后从layer_out中提取最大值的位置，最后输出位置的y坐标
            minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
            return predict_maxLoc.y;
        }
        else
        {
            std::cout << "Please give one sample alone and ensure input.rows = layer[0].rows" << std::endl;
            return -1;
        }
    }

    /**
     * 预测多条数据，循环调用predict_one
     * @param input
     * @return
     */
    std::vector<int> libNet::predict(cv::Mat &input)
    {
        std::vector<int> predicted_labels;
        if (input.rows == (layer[0].rows) && input.cols > 1)
        {
            for (int i = 0; i < input.cols; ++i)
            {
                cv::Mat sample = input.col(i);
                int predicted_label = predict_one(sample);
                predicted_labels.push_back(predicted_label);
            }
        }
        return predicted_labels;
    }

    /**
     * 保存模型，模型的主要参数：
     * 1. layer_neuron_num，各层神经元数目，这是生成神经网络需要的唯一参数
     * 2. weights，神经网络初始化之后需要用训练好的权值矩阵去初始化权值
     * 3. activation_function，使用神经网络的过程其实就是前向计算的过程，显然需要知道激活函数是什么
     * 4. learning_rate，如果要在现有模型的基础上继续训练以得到更好的模型，更新权值的时候需要用到这个函数
     * @param filename
     */
    void libNet::save(std::string filename)
    {
        cv::FileStorage model(filename, cv::FileStorage::WRITE);
        model << "layer_neuron_num" << layer_neuron_num;
        model << "learning_rate" << learning_rate;
        model << "activation_function" << activation_function;

        for (int i = 0; i < weights.size(); i++)
        {
            std::string weight_name = "weight_" + std::to_string(i);
            model << weight_name << weights[i];
        }
        model.release();
    }

    // 加载模型
    void libNet::load(std::string filename)
    {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_, target_;

        fs["layer_neuron_num"] >> layer_neuron_num;
        initNet(layer_neuron_num);

        for (int i = 0; i < weights.size(); i++)
        {
            std::string weight_name = "weight_" + std::to_string(i);
            fs[weight_name] >> weights[i];
        }

        fs["learning_rate"] >> learning_rate;
        fs["activation_function"] >> activation_function;

        fs.release();
    }

    /**
     * 读取数据
     * @param filename 数据文件
     * @param input 未初始化的输入的标签
     * @param label 未初始化的目标
     * @param sample_num 数量
     * @param start 开始位置
     */
    void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start)
    {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_;
        cv::Mat target_;
        fs["input"] >> input_;
        fs["target"] >> target_;
        fs.release();
        input = input_(cv::Rect(start, 0, sample_num, input_.rows));
        label = target_(cv::Rect(start, 0, sample_num, target_.rows));
    }

    //绘制损失函数
    void draw_curve(cv::Mat& board, std::vector<double> points)
    {
        cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
        board = board_;
        cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
        cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

        for (size_t i = 0; i < points.size() - 1; i++)
        {
            cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
            cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
            cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
            if (i >= 1000)
            {
                return;
            }
        }
        cv::imshow("Loss", board);
        cv::waitKey(1);
    }
} // CuiQin