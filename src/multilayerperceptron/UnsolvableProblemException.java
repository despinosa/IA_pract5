/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

/**
 *
 * @author inesita
 */
class UnsolvableProblemException extends Exception {
    private String message;

    public UnsolvableProblemException(Throwable throwable) {
        super(throwable);
        this.message = "No se puede resolver el problema dentro del margen" +
                "de error especificado. Intenta con un margen más grande.";
    }
    
}
