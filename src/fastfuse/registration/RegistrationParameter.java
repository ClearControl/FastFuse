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

  public void setMaxNumberOfEvaluations(int pMaxNumberOfEvaluations);

  public float getScaleZ();

  public void setScaleZ(float pScaleZ);

  public double[] getLowerBounds();

  public void setLowerBounds(double[] pLowerBound);

  public double[] getUpperBounds();

  public void setUpperBounds(double[] pUpperBound);

  public double[] getInitialTransformation();

  public void setZeroTransformMatrix(Matrix4f pZeroTransformMatrix);

  public Matrix4f getZeroTransformMatrix();

  public void setInitialTransformation(double... theta);

  public int getNumberOfRestarts();

  public void setNumberOfRestarts(int pRestarts);

  public default int getOpenCLReductionThreshold()
  {
    return getOpenCLGroupSize();
  }

}
