/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

/**
 *
 * @author daniel
 */
public class UnexpectedValueException extends Exception {

    private Boolean inconsistency;
    private Integer recordNotFound;

    public UnexpectedValueException(Boolean wrongSize,
            Integer recordNotFound) {
        super("El dato de entrada no es válido");
        this.wrongSize = wrongSize;
        this.recordNotFound = recordNotFound;
    }

    UnexpectedValueException() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public Boolean getWrongSize () {
        return wrongSize;
    }
    
    public Integer getRecordNotFound () {
        return recordNotFound;
    }
}
