package multilayerperceptron;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author daniel
 */
abstract class Neuron {
    /* Neurona de la red. Pueden ser de entrada, salida o escondidas.
     * Cada una está asociada con todas las neuronas del siguiente nivel.
     * 
     * expectedValue: Valor esperado para las neuronas de salida. null para
     *     las demás.
     * value: Valor en la neurona. Es null hasta que se calcula o introduce.
     * error: Error en las neuronas de salida o de la capa escondida. Es
     *     null hasta que se calcula.
     * previousNeurons: Mapeo de las neurones anteriores a la actual
     *     (down-top) con sus pesos de enlace. No existe para las
     *     neuronas de entrada.
     * nextNeurons: Mapeo de las neurones siguientes a la actual (top-down)
     *     con sus pesos de enlace. No existe para las neuronas de
     *     salida.
     * weightsError: Mapeo de las neuronas siguientes a la actual (top-down)
     *     con su error en el peso de enlace correspondiente. Siempre es
    .*     null para las neuronas de salida.
     */
    Double value;
    
    abstract public Double getValue();
}

class InputNeuron extends Neuron {
    Map<NonInputNeuron, Link> nextNeurons;

    public void setValue (double value) {
        this.value = value;
    }

    public Double getValue () {
        return value;
    }

    public InputNeuron () {
        nextNeurons = new HashMap<>();
    }

    private void addNextNeuron (NonInputNeuron next, Link link) {
        nextNeurons.put(next, link);
    }

    public void link (NonInputNeuron next) {
        Link link = new Link();
        this.addNextNeuron(next, link);
        next.addPreviousNeuron(this, link);
    }
}

abstract class NonInputNeuron extends Neuron {
    Map<Neuron, Link> previousNeurons;
    Double error;

    public double getError ()  throws NullPointerException {
        if (error != null) {
            return error;
        } else {
            calculateError();
            return error;
        }
    }
    protected void addPreviousNeuron(Neuron previous, Link link) {
        previousNeurons.put(previous, link);
    }

    protected double net () throws NullPointerException {
        double rtrn = 0.0;
        for (Neuron previousNeuron : previousNeurons.keySet()) {
            rtrn += previousNeuron.value *
                    previousNeurons.get(previousNeuron).getWeight();
        }
        return rtrn;
    }

    protected double sigma () {
        return 1 / (1 + exp(-net()));
    }

    protected double primeSigma () {
        if (value == null) calculateValue();
        return value * (1 - value);
    }

    public void calculateValue () {
        value = sigma();
    }

    abstract public void calculateError ();

    public void updateWeights () {
        for (Neuron previousNeuron : previousNeurons.keySet()) {
            previousNeurons.get(previousNeuron).updateWeight();
        }
    }
}

class HiddenNeuron extends NonInputNeuron {
    Map<NonInputNeuron, Link> nextNeurons;

    public HiddenNeuron () {
        previousNeurons = new HashMap<>();
        nextNeurons = new HashMap<>();
    }

    private void addNextNeuron (NonInputNeuron next, Link link) {
        nextNeurons.put(next, link);
    }

    public void link (NonInputNeuron next) {
        Link link = new Link();
        this.addNextNeuron(next, link);
        next.addPreviousNeuron(this, link);
    }

    public void calculateError () {
        error = 0.0;
        for (NonInputNeuron nextNeuron : nextNeurons.keySet()) {
            error += nextNeuron.error *
                     nextNeurons.get(nextNeuron).getWeight();
        }
        error *= primeSigma();
    }

    public void calculateWeightsError (double learningRate)  throws NullPointerException {
        for (Neuron previousNeuron : previousNeurons.keySet()) {
            double linkError = learningRate * error * previousNeuron.value;
            previousNeurons.get(previousNeuron).setError(linkError);
        }
    }

    public Double getValue () {
        return value;
    }
}



class OutputNeuron extends NonInputNeuron {
    Double expectedValue;
    Map<Neuron, Link> previousNeurons;

    public OutputNeuron () {
        previousNeurons = new HashMap<>();
    }

    public OutputNeuron (double expectedValue) {
        // neuronas de entrada o salida
        this.expectedValue = expectedValue;
        previousNeurons = new HashMap<>();
    }

    public void calculateError () {
        error = (expectedValue - value) * primeSigma();
    }

    protected double net () throws NullPointerException {
        double rtrn = 0.0;
        for (Neuron previousNeuron : previousNeurons.keySet()) {
            rtrn += previousNeuron.value *
                    previousNeurons.get(previousNeuron).getWeight();
        }
        return rtrn;
    }

    public boolean evaluate (double threshold) throws
            UnexpectedActionException {
        return abs(value-expectedValue) < threshold;
    }

    public Double getValue() {
        if (value == null) calculateValue();
        return value;
    }

    public void calculateWeightsError (double learningRate)  throws NullPointerException {
        for (Neuron previousNeuron : previousNeurons.keySet()) {
            double linkError = learningRate * error * previousNeuron.value;
            previousNeurons.get(previousNeuron).setError(linkError);
        }
    }
}

class Link {
    private double weight;
    private Double error;

    public Link() {}

    public Link(double weight) {
        this.weight = weight;
    }

    public void updateWeight () {
        weight += error;
        error = 0.0;
    }

    public double getWeight () {
        return weight;
    }

    public void setError (double error) {
        this.error = error;
    }
}
