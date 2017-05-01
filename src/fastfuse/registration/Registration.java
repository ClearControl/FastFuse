package fastfuse.registration;

import javax.vecmath.Matrix4f;

/**
 * Stack registration
 *
 * @author royer
 */
public class Registration
{
  private Matrix4f mInitialMatrix;

  private int mMaxNumberOfIterations;

  /**
   * Sets the initial matrix used for registration
   * 
   * @param pInitialMatrix
   *          initial matrix
   */
  public void setInitialTransformMatrix(Matrix4f pInitialMatrix)
  {
    mInitialMatrix = pInitialMatrix;
  }

  /**
   * Sets max number of iterations
   * 
   * @param pMaxNumberOfIterations
   *          maximal numbr of iterations
   */
  public void setMaxNumberOfIterations(int pMaxNumberOfIterations)
  {
    mMaxNumberOfIterations = pMaxNumberOfIterations;
  }

  public void register()
  {
    // TODO Auto-generated method stub

  }
}
