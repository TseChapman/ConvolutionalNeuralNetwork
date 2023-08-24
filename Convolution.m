classdef Convolution
    properties
        inputSize % number of input to the fully connected layer

        outputSize % number of class to classify

        numNeuron % number of neurons in the hidden and output layers

        kernelSize % the kernel size for running convolution

        numFilters % number of filters use for running convolution

        kernels % randomized 3D matrix, kernelSize x kernelSize x numFilters

        hiddenWeight % randomized hidden layer weight

        outputWeight % randomized output layer weight

        lastKernels % used to record the kernels from previous epoch, used in variable learning rate

        lastHiddenWeight % used to record the hidden layer weight from previous epoch, used in variable learning rate

        lastOutputWeight % used to record the output layer weight from previous epoch, used in variable learning rate

        rho % decay coefficient

        zeta % variable learning rate threshold

        eta % growth coefficient

        learningRate % learning rate for backpropagation
    end
    methods
        % Constructor
        % Take the input an initialize the properties
        function obj = Convolution(inputSize, outputSize, numNeuron, kernelSize, numFilters, learningRate, rho,zeta,eta)
            obj.inputSize = inputSize;
            obj.outputSize = outputSize;
            obj.numNeuron = numNeuron;
            obj.kernelSize = kernelSize;
            obj.numFilters = numFilters;
            rng(1);
            obj.kernels = 1e-2*randn([kernelSize kernelSize numFilters]);
            obj.hiddenWeight = (2*rand(obj.numNeuron, obj.inputSize) - 1) * sqrt(6) / sqrt(360 + obj.inputSize);
            obj.outputWeight = (2*rand(obj.outputSize, obj.numNeuron) - 1) * sqrt(6) / sqrt(10 + obj.numNeuron);

            obj.rho = rho;
            obj.zeta = zeta;
            obj.eta = eta;

            obj.learningRate = learningRate;
        end

        % forward
        % Produce a classification output using network's weights and the
        %   inputted image
        function [in2, in3, in4, wp2, a] = forward(obj, image)
            in1 = obj.conv(image,obj.kernels);
            in2 = obj.ReLU(in1);
            in3 = obj.pool(in2);
            in4 = reshape(in3, [], 1);
            wp1 = obj.hiddenWeight * in4;
            wp2 = obj.ReLU(wp1);
            n = obj.outputWeight * wp2;
            a = obj.Softmax(n);
        end

        % trainTest
        % Perform network training and validation
        % record MSE per epoch and graph the result
        function obj = trainTest(obj, inputs, outputs, epochs, varyLearningRate, testingSet, testingLabels)
            epochMse = zeros(1,epochs);
            testMSE = zeros(1,epochs);
            accuracies = zeros(1,epochs);
            for epoch = 1:epochs
                disp("Begin Epoch " + epoch)
                % train the model per epoch
                [obj, accuracy, trainingMse] = obj.train(inputs,outputs);
                accuracies(epoch) = accuracy;
                epochMse(epoch) = trainingMse;
                disp("Training accuracy: " + accuracy)
                disp("Training MSE: " + trainingMse)

                if varyLearningRate==1 && epoch~=1
                    obj = obj.varyLearningRate(epoch,epochMse);
                end
                
                % validate the trained network
                testMSE(epoch) = obj.test(testingSet, testingLabels);
                disp("Testing MSE: " + testMSE(epoch))
            end
            
            % graph the result MSE
            figure; hold on 
            a1 = plot(epochMse); M1 = 'Training MSE'; 
            a2 = plot(testMSE); M2 = 'Testing MSE'; 
            legend([a1;a2], M1, M2); 
            hold off 
            f_title = ['CNN Errors ' num2str(obj.numNeuron) ' Neurons ' num2str(epochs) ' Epochs ' num2str(obj.kernelSize) ' Kernel Size ' num2str(obj.numFilters) ' Number of Filters'];
            title(f_title) 
            xlabel('Number of Epochs') 
            ylabel('Mean Squared Error') 

            figure;
            plot(accuracies)
            title('Network Performance Throughout Training') 
            xlabel('Epochs') 
            ylabel('Test Accuracy')
        end
    end
    methods (Access = private)
        % test
        % Run validation and record the MSE
        function testMSE = test(obj, testingSet, testingLabels)
            testMSE = 0.0;
            N = length(testingLabels);
            for i = 1:N
                image = testingSet(:,:,i);
                output = zeros(obj.outputSize, 1);
                output(sub2ind(size(output), testingLabels(i), 1)) = 1;

                [in2, in3, in4, wp2, a] = obj.forward(image);
                err = obj.error(output, a);
                testMSE = testMSE + sum(err.^2);
            end
            testMSE = testMSE / N;
        end

        % conv
        % Convolution layer
        function output = conv(~, image, kernels)
            [krow, kcol, numFilter] = size(kernels);
            [irow, icol] = size(image);

            outRow = irow - krow + 1;
            outCol = icol - kcol + 1;

            output = zeros(outRow, outCol, numFilter);

            for f = 1:numFilter
                filter = kernels(:,:,f);
                filter = rot90(squeeze(filter), 2);
                output(:,:,f) = conv2(image,filter,'valid');
            end
        end

        % error
        % Calculate loss
        function err = error(~, t, a)
            err = t - a;
        end

        % ReLU
        % rectified Linear Unit (activation function)
        function output = ReLU(~,image)
            output = max(0,image);
        end

        % Softmax
        % probability distribution (activation function)
        function output = Softmax(~,image)
            ex = exp(image);
            output = ex / sum(ex);
        end

        % pool
        % Mean pooling layer
        function output = pool(~, image)
            [irow, icol, numFilter] = size(image);
            output = zeros(irow/2, icol/2, numFilter);
            for f = 1:numFilter
                filter = ones(2) / (2*2);
                temp = conv2(image(:,:,f), filter, 'valid');
                output(:,:,f) = temp(1:2:end, 1:2:end);
            end
        end

        % train
        % train the network in batches and in each batch calculate the loss
        % and update the weight accordingly
        function [obj, accuracy, trainingMse] = train(obj,inputs,outputs)
            beta = 0.95;

            momentum1 = zeros(size(obj.kernels));
            momentum5 = zeros(size(obj.hiddenWeight));
            momentumo = zeros(size(obj.outputWeight));

            N = length(outputs);
            accuracy = 0;
            trainingMse = 0;

            bsize = 100; % batch size
            blist = 1:bsize:(N-bsize+1);

            for batch = 1:length(blist)
                dKernel = zeros(size(obj.kernels));
                dHidden = zeros(size(obj.hiddenWeight));
                dOutput = zeros(size(obj.outputWeight));

                begin = blist(batch);
                for i = begin:begin+bsize-1
                    image = inputs(:,:,i);
                    output = zeros(obj.outputSize, 1);
                    output(sub2ind(size(output), outputs(i), 1)) = 1;
                    
                    % classify the image and get the loss from true value
                    [in2, in3, in4, wp2, a] = obj.forward(image);
                    err = obj.error(output, a);
                    [~, temp] = max(a);
                    if (outputs(i) == temp)
                        accuracy = accuracy + 1;
                    end
                    trainingMse = trainingMse + sum(err.^2);

                    % update the weight derivative
                    [dKernel, dHidden, dOutput, obj] = obj.weightBiasUpdate(err, in2, in3, in4, wp2,dKernel, dHidden, dOutput, image);
                end

                dKernel = dKernel / bsize;
                dHidden = dHidden / bsize;
                dOutput = dOutput / bsize;

                % Update the weights and kernels
                momentum1 = obj.learningRate * dKernel + beta * momentum1;
                obj.kernels = obj.kernels + momentum1;

                momentum5 = obj.learningRate * dHidden + beta * momentum5;
                obj.hiddenWeight = obj.hiddenWeight + momentum5;

                momentumo = obj.learningRate * dOutput + beta * momentumo;
                obj.outputWeight = obj.outputWeight + momentumo;
            end
            accuracy = accuracy / N;
            trainingMse = trainingMse / N;
        end

        % weightBiasUpdate
        % calculate the derivatives for the kernels, hidden layer's weight,
        %   and the output layer's weight
        function [dKernel, dHidden, dOutput, obj] = weightBiasUpdate( ...
            obj, err, in2, in3, in4, wp2,dKernel, dHidden, dOutput, image)
            del1 = err;
            e5 = obj.outputWeight' * del1;
            del2 = (wp2 > 0) .* e5;
            e4 = obj.hiddenWeight' * del2;
            e3 = reshape(e4, size(in3));
            e2 = zeros(size(in2));
            weight = ones(size(in2)) / (2*2);
            
            for c = 1:20
                e2(:,:,c) = kron(e3(:,:,c), ones([2 2])) .* weight(:,:,c);
            end

            delta2 = (in2 > 0) .* e2;
            delta1 = zeros(size(obj.kernels));

            for c = 1:20
                delta1(:,:,c) = conv2(image(:,:), rot90(delta2(:,:,c), 2), 'valid');
            end

            dKernel = dKernel + delta1;
            dHidden = dHidden + del2 * in4';
            dOutput = dOutput + del1 * wp2';
        end

        % varyLearningRate
        % update the learning rate based on the MSE difference between
        %   current and previous epoch
        function obj = varyLearningRate(obj,epo,trainingMse)

            % calculate %change in MSE since last epoch
            changeInMSE = trainingMse(epo)/trainingMse(epo-1);
            fprintf("change in mse: %f\n",changeInMSE);
            
            % good gradient, go faster
            if changeInMSE < 1
                disp("increasing learning rate");
                obj.learningRate = obj.learningRate*obj.eta;

            % ok gradient, go same speed
            elseif changeInMSE < obj.zeta
                disp("not changing learning rate");
            
            % bad gradient, try again but slower
            else
                disp("decreasing learning rate");
                obj.kernels = obj.lastKernels;
                obj.hiddenWeight = obj.lastHiddenWeight;
                obj.outputWeight = obj.lastOutputWeight;
                obj.learningRate = obj.learningRate*obj.rho;
            end

            % save weights after update
            obj.lastKernels = obj.kernels;
            obj.lastHiddenWeight = obj.hiddenWeight;
            obj.lastOutputWeight = obj.outputWeight;
        end
    end
end