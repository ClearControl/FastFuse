package fastfuse.registration;

import javax.vecmath.Matrix4f;

/**
 * Interface for registration parameter
 * 
 * @author uschmidt
 */

public interface RegistrationParameter
{

  public Class<?> getClassForKernelBasePath();

  public String getKernelSourceFile();

  public int getOpenCLGroupSize();

  public double[] perturbTransformation(double... theta);

  public boolean getWaitToFinish();

  public int getMaxNumberOfEvaluations();

  public float getScaleZ();

  public double[] getLowerBounds();

  public double[] getUpperBounds();

  public double[] getInitialTransformation();

  public Matrix4f getZeroTransformMatrix();

  public int getNumberOfRestarts();

  public default int getOpenCLReductionThreshold()
  {
    return getOpenCLGroupSize();
  }

}
