package multilayerperceptron;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author daniel
 */
public class MultilayerPerceptron {
    /* Objeto red neuronal multicapa de perceptrones con entrenamiento
     * backpropagation. Dada una base de conocimiento genera una red neuronal
     * que resuelva el problema representado. Puede también inicializarse con
     * una red ya entrenada representada en un archivo DOT.
     * 
     * inputs: Neuronas de entrada. No se les puede calcular error.
     * hiddenLayers: Capas de neuronas escondidas.
     * outputs: Neuronas de salida. Consideran un valor esperado fijo.
     * threshold: Umbral o tolerancia en la lectura de los valores de salida.
     * learningRate: Tasa de aprendizaje en el entrenamiento con
     *     backpropagation.
     * maxGlobalError: Máximo error cuadrático medio global aceptado como
     *     solución.
     * knowledgeBase: Base de conocimientos empleada para crear la red. Consiste
     *     en mapeos patrón -> decisión.
     */

    InputNeuron[] inputs;
    HiddenNeuron[][] hiddenLayers;
    OutputNeuron[] outputs;
    double threshold;
    Double learningRate;
    Map<Double[], Double[]> knowledgeBase;

    public Map<Double[], Double[]> getKnowledgeBase() {
        return knowledgeBase;
    }

    public void setKnowledgeBase(Map<Double[], Double[]> knowledgeBase) {
        this.knowledgeBase = knowledgeBase;
    }

    public MultilayerPerceptron(int inputs, int hiddenLayers, int neuronsPerLayer,
            int outputs, double threshold, double learningRate,
            Map<Double[], Double[]> knowledgeBase) {
        this.inputs = new InputNeuron[inputs];
        for (int i = 0; i < inputs; i++) {
            this.inputs[i] = new InputNeuron();
        }
        this.hiddenLayers = new HiddenNeuron[hiddenLayers][neuronsPerLayer];
        for (int i = 1; i < hiddenLayers; i++) {
            for (int j = 0; j < neuronsPerLayer; j++) {
                this.hiddenLayers[i][j] = new HiddenNeuron();
            }
        }
        this.outputs = new OutputNeuron[outputs];
        for (int i = 0; i < outputs; i++) {
            this.outputs[i] = new OutputNeuron();
        }
        for (HiddenNeuron hidden : this.hiddenLayers[0]) {
            for (InputNeuron input : this.inputs) {
                input.link(hidden);
            }
        }
        for (int i = 1; i < hiddenLayers; i++) {
            for (int j = 0; j < neuronsPerLayer; j++) {
                for (HiddenNeuron hidden : this.hiddenLayers[i-1]) {
                    hidden.link(this.hiddenLayers[i][j]);
                }
            }
        }
        for (HiddenNeuron hidden : this.hiddenLayers[hiddenLayers-1]) {
            for (OutputNeuron output : this.outputs) {
                hidden.link(output);
            }
        }
        this.threshold = threshold;
        this.learningRate = learningRate;
        this.knowledgeBase = knowledgeBase;
    }

    protected double iterate() throws UnexpectedActionException {
        double[] squaredMeanErrors = new double[outputs.length];
        for (double squaredMeanError : squaredMeanErrors) {
            squaredMeanError = 0.0;
        }
        for (Double[] inputValue : knowledgeBase.keySet()) {
            for (int i = 0; i < inputs.length; i++) {
                inputs[i].setValue((double) inputValue[i]);
            }
            for (int i = 0; i < hiddenLayers.length; i++) {
                for (int j = 0; j < hiddenLayers[i].length; j++) {
                    hiddenLayers[i][j].calculateValue();
                }
            }
            boolean allOutputsOk;
            for (OutputNeuron output : outputs) {
                output.calculateValue();
                allOutputsOk = output.evaluate(threshold);
            }
        }
        double globalError = 0.0;
        for (double squaredMeanError : squaredMeanErrors) {
            globalError += squaredMeanError;
        }
        return globalError / (double) squaredMeanErrors.length;
    }

    protected void backpropagation() {
        for (OutputNeuron output : outputs) {
            output.calculateError();
            output.calculateWeightsError(learningRate);
        }
        for (int i = hiddenLayers.length; i > 0; i++) {
            for (HiddenNeuron hidden : hiddenLayers[i]) {
                hidden.calculateError();
                hidden.calculateWeightsError(learningRate);
            }
        }
        for (OutputNeuron output : outputs) {
            output.calculateError();
            output.calculateWeightsError(learningRate);
        }
        for (int i = hiddenLayers.length; i > 0; i++) {
            for (HiddenNeuron hidden : hiddenLayers[i]) {
                hidden.calculateError();
                hidden.calculateWeightsError(learningRate);
            }
        }
    }

    public void train(double maxGlobalError) throws UnsolvableProblemException {
        try {
            double currentError;
            do {
                currentError = iterate();
            } while (currentError > maxGlobalError);
        } catch (StackOverflowError soe) {
            throw new UnsolvableProblemException(soe);
        } catch (UnexpectedActionException ex) {
            Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /*
     public void init () throws UnexpectedActionException {
     try {
     for(int i = 0; i < hiddenLayers.length; i++){
     for(Neuron hidden : hiddenLayers[i]) {
     hidden.calculateValue();
     }
     }
     double globalError = 0.0;
     for (Neuron output : outputs) {
     output.calculateValue();
     output.calculateError();
     globalError +=
     }
     } catch (NullPointerException npe) {
     throw new UnexpectedActionException(npe);
     }
     }
     */
    public static void main(String[] args) {
        Map<Double[], Double[]> kb = new HashMap<>();
        List<Double[]> values = new ArrayList<>();
        values.add(new Double[]{5.1, 3.5, 1.4, 0.1});
        values.add(new Double[]{4.9, 3.0, 1.4, 0.2});
        values.add(new Double[]{4.7, 3.2, 1.3, 0.2});
        values.add(new Double[]{4.6, 3.1, 1.5, 0.2});
        values.add(new Double[]{5.0, 3.6, 1.4, 0.2});
        values.add(new Double[]{5.4, 3.9, 1.7, 0.4});
        values.add(new Double[]{4.6, 3.4, 1.4, 0.3});
        values.add(new Double[]{5.0, 3.4, 1.5, 0.2});
        values.add(new Double[]{4.4, 2.9, 1.4, 0.2});
        values.add(new Double[]{4.9, 3.1, 1.5, 0.1});
        values.add(new Double[]{5.4, 3.7, 1.5, 0.2});
        values.add(new Double[]{4.8, 3.4, 1.6, 0.2});
        values.add(new Double[]{4.8, 3.0, 1.4, 0.1});
        values.add(new Double[]{4.3, 3.0, 1.1, 0.1});
        values.add(new Double[]{5.8, 4.0, 1.2, 0.2});
        values.add(new Double[]{5.7, 4.4, 1.5, 0.4});
        values.add(new Double[]{5.4, 3.9, 1.3, 0.4});
        values.add(new Double[]{5.1, 3.5, 1.4, 0.3});
        values.add(new Double[]{5.7, 3.8, 1.7, 0.3});
        values.add(new Double[]{5.1, 3.8, 1.5, 0.3});
        values.add(new Double[]{5.4, 3.4, 1.7, 0.2});
        values.add(new Double[]{5.1, 3.7, 1.5, 0.4});
        values.add(new Double[]{4.6, 3.6, 1.0, 0.2});
        values.add(new Double[]{5.1, 3.3, 1.7, 0.5});
        values.add(new Double[]{4.8, 3.4, 1.9, 0.2});
        values.add(new Double[]{5.0, 3.0, 1.6, 0.2});
        values.add(new Double[]{5.0, 3.4, 1.6, 0.4});
        values.add(new Double[]{5.2, 3.5, 1.5, 0.2});
        values.add(new Double[]{5.2, 3.4, 1.4, 0.2});
        values.add(new Double[]{4.7, 3.2, 1.6, 0.2});
        values.add(new Double[]{4.8, 3.1, 1.6, 0.2});
        values.add(new Double[]{5.4, 3.4, 1.5, 0.4});
        values.add(new Double[]{5.2, 4.1, 1.5, 0.1});
        values.add(new Double[]{5.5, 4.2, 1.4, 0.2});
        values.add(new Double[]{4.9, 3.1, 1.5, 0.2});
        values.add(new Double[]{5.0, 3.2, 1.2, 0.2});
        values.add(new Double[]{5.5, 3.5, 1.3, 0.2});
        values.add(new Double[]{4.9, 3.6, 1.4, 0.1});
        values.add(new Double[]{4.4, 3.0, 1.3, 0.2});
        values.add(new Double[]{5.1, 3.4, 1.5, 0.2});
        values.add(new Double[]{5.0, 3.5, 1.3, 0.3});
        values.add(new Double[]{4.5, 2.3, 1.3, 0.3});
        values.add(new Double[]{4.4, 3.2, 1.3, 0.2});
        values.add(new Double[]{5.0, 3.5, 1.6, 0.6});
        values.add(new Double[]{5.1, 3.8, 1.9, 0.4});
        values.add(new Double[]{4.8, 3.0, 1.4, 0.3});
        values.add(new Double[]{5.1, 3.8, 1.6, 0.2});
        values.add(new Double[]{4.6, 3.2, 1.4, 0.2});
        values.add(new Double[]{5.3, 3.7, 1.5, 0.2});
        values.add(new Double[]{5.0, 3.3, 1.4, 0.2});
        values.add(new Double[]{7.0, 3.2, 4.7, 1.4});
        values.add(new Double[]{6.4, 3.2, 4.5, 1.5});
        values.add(new Double[]{6.9, 3.1, 4.9, 1.5});
        values.add(new Double[]{5.5, 2.3, 4.0, 1.3});
        values.add(new Double[]{6.5, 2.8, 4.6, 1.5});
        values.add(new Double[]{5.7, 2.8, 4.5, 1.3});
        values.add(new Double[]{6.3, 3.3, 4.7, 1.6});
        values.add(new Double[]{4.9, 2.4, 3.3, 1.0});
        values.add(new Double[]{6.6, 2.9, 4.6, 1.3});
        values.add(new Double[]{5.2, 2.7, 3.9, 1.4});
        values.add(new Double[]{5.0, 2.0, 3.5, 1.0});
        values.add(new Double[]{5.9, 3.0, 4.2, 1.5});
        values.add(new Double[]{6.0, 2.2, 4.0, 1.0});
        values.add(new Double[]{6.1, 2.9, 4.7, 1.4});
        values.add(new Double[]{5.6, 2.9, 3.6, 1.3});
        values.add(new Double[]{6.7, 3.1, 4.4, 1.4});
        values.add(new Double[]{5.6, 3.0, 4.5, 1.5});
        values.add(new Double[]{5.8, 2.7, 4.1, 1.0});
        values.add(new Double[]{6.2, 2.2, 4.5, 1.5});
        values.add(new Double[]{5.6, 2.5, 3.9, 1.1});
        values.add(new Double[]{5.9, 3.2, 4.8, 1.8});
        values.add(new Double[]{6.1, 2.8, 4.0, 1.3});
        values.add(new Double[]{6.3, 2.5, 4.9, 1.5});
        values.add(new Double[]{6.1, 2.8, 4.7, 1.2});
        values.add(new Double[]{6.4, 2.9, 4.3, 1.3});
        values.add(new Double[]{6.6, 3.0, 4.4, 1.4});
        values.add(new Double[]{6.8, 2.8, 4.8, 1.4});
        values.add(new Double[]{6.7, 3.0, 5.0, 1.7});
        values.add(new Double[]{6.0, 2.9, 4.5, 1.5});
        values.add(new Double[]{5.7, 2.6, 3.5, 1.0});
        values.add(new Double[]{5.5, 2.4, 3.8, 1.1});
        values.add(new Double[]{5.5, 2.4, 3.7, 1.0});
        values.add(new Double[]{5.8, 2.7, 3.9, 1.2});
        values.add(new Double[]{6.0, 2.7, 5.1, 1.6});
        values.add(new Double[]{5.4, 3.0, 4.5, 1.5});
        values.add(new Double[]{6.0, 3.4, 4.5, 1.6});
        values.add(new Double[]{6.7, 3.1, 4.7, 1.5});
        values.add(new Double[]{6.3, 2.3, 4.4, 1.3});
        values.add(new Double[]{5.6, 3.0, 4.1, 1.3});
        values.add(new Double[]{5.5, 2.5, 4.0, 1.3});
        values.add(new Double[]{5.5, 2.6, 4.4, 1.2});
        values.add(new Double[]{6.1, 3.0, 4.6, 1.4});
        values.add(new Double[]{5.8, 2.6, 4.0, 1.2});
        values.add(new Double[]{5.0, 2.3, 3.3, 1.0});
        values.add(new Double[]{5.6, 2.7, 4.2, 1.3});
        values.add(new Double[]{5.7, 3.0, 4.2, 1.2});
        values.add(new Double[]{5.7, 2.9, 4.2, 1.3});
        values.add(new Double[]{6.2, 2.9, 4.3, 1.3});
        values.add(new Double[]{5.1, 2.5, 3.0, 1.1});
        values.add(new Double[]{5.7, 2.8, 4.1, 1.3});
        values.add(new Double[]{6.3, 3.3, 6.0, 2.5});
        values.add(new Double[]{5.8, 2.7, 5.1, 1.9});
        values.add(new Double[]{7.1, 3.0, 5.9, 2.1});
        values.add(new Double[]{6.3, 2.9, 5.6, 1.8});
        values.add(new Double[]{6.5, 3.0, 5.8, 2.2});
        values.add(new Double[]{7.6, 3.0, 6.6, 2.1});
        values.add(new Double[]{4.9, 2.5, 4.5, 1.7});
        values.add(new Double[]{7.3, 2.9, 6.3, 1.8});
        values.add(new Double[]{6.7, 2.5, 5.8, 1.8});
        values.add(new Double[]{7.2, 3.6, 6.1, 2.5});
        values.add(new Double[]{6.5, 3.2, 5.1, 2.0});
        values.add(new Double[]{6.4, 2.7, 5.3, 1.9});
        values.add(new Double[]{6.8, 3.0, 5.5, 2.1});
        values.add(new Double[]{5.7, 2.5, 5.0, 2.0});
        values.add(new Double[]{5.8, 2.8, 5.1, 2.4});
        values.add(new Double[]{6.4, 3.2, 5.3, 2.3});
        values.add(new Double[]{6.5, 3.0, 5.5, 1.8});
        values.add(new Double[]{7.7, 3.8, 6.7, 2.2});
        values.add(new Double[]{7.7, 2.6, 6.9, 2.3});
        values.add(new Double[]{6.0, 2.2, 5.0, 1.5});
        values.add(new Double[]{6.9, 3.2, 5.7, 2.3});
        values.add(new Double[]{5.6, 2.8, 4.9, 2.0});
        values.add(new Double[]{7.7, 2.8, 6.7, 2.0});
        values.add(new Double[]{6.3, 2.7, 4.9, 1.8});
        values.add(new Double[]{6.7, 3.3, 5.7, 2.1});
        values.add(new Double[]{7.2, 3.2, 6.0, 1.8});
        values.add(new Double[]{6.2, 2.8, 4.8, 1.8});
        values.add(new Double[]{6.1, 3.0, 4.9, 1.8});
        values.add(new Double[]{6.4, 2.8, 5.6, 2.1});
        values.add(new Double[]{7.2, 3.0, 5.8, 1.6});
        values.add(new Double[]{7.4, 2.8, 6.1, 1.9});
        values.add(new Double[]{7.9, 3.8, 6.4, 2.0});
        values.add(new Double[]{6.4, 2.8, 5.6, 2.2});
        values.add(new Double[]{6.3, 2.8, 5.1, 1.5});
        values.add(new Double[]{6.1, 2.6, 5.6, 1.4});
        values.add(new Double[]{7.7, 3.0, 6.1, 2.3});
        values.add(new Double[]{6.3, 3.4, 5.6, 2.4});
        values.add(new Double[]{6.4, 3.1, 5.5, 1.8});
        values.add(new Double[]{6.0, 3.0, 4.8, 1.8});
        values.add(new Double[]{6.9, 3.1, 5.4, 2.1});
        values.add(new Double[]{6.7, 3.1, 5.6, 2.4});
        values.add(new Double[]{6.9, 3.1, 5.1, 2.3});
        values.add(new Double[]{5.8, 2.7, 5.1, 1.9});
        values.add(new Double[]{6.8, 3.2, 5.9, 2.3});
        values.add(new Double[]{6.7, 3.3, 5.7, 2.5});
        values.add(new Double[]{6.7, 3.0, 5.2, 2.3});
        values.add(new Double[]{6.3, 2.5, 5.0, 1.9});
        values.add(new Double[]{6.5, 3.0, 5.2, 2.0});
        values.add(new Double[]{6.2, 3.4, 5.4, 2.3});
        values.add(new Double[]{5.9, 3.0, 5.1, 1.8});
        List<Double[]> decisions = new ArrayList<>();
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{1.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{2.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        decisions.add(new Double[]{3.0});
        if (decisions.size() == values.size()) {
            for (int i = 0; i < decisions.size(); i++) {
                kb.put(values.get(i), decisions.get(i));
            }
        }
        MultilayerPerceptron network = new MultilayerPerceptron(4, 2, 4, 1, 0.5,
            0.25, kb);
        try {
            network.train(0.2);
        } catch (UnsolvableProblemException ex) {
            System.out.println(ex.getMessage());
        }
    }
}
