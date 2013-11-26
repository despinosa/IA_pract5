/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

/**
 *
 * @author daniel
 */
public class UnexpectedActionException extends Exception {
    public UnexpectedActionException() {
        super("No se esperaba esta acción");
    }
    
    public UnexpectedActionException(Throwable t) {
        super("No se esperaba esta acción", t);
    }
}
