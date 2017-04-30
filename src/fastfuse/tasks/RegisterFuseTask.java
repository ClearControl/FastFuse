package fastfuse.tasks;

import java.util.Set;

import fastfuse.FastFusionEngineInterface;

/**
 * Registered fusion task
 *
 * @author royer
 */
public class RegisterFuseTask implements FusionTaskInterface
{

  public RegisterFuseTask(String pString,
                          String pString2,
                          String pString3)
  {
    // TODO Auto-generated constructor stub
  }

  @Override
  public boolean checkIfRequiredImagesAvailable(Set<String> pAvailableImagesKeys)
  {
    // TODO Auto-generated method stub
    return false;
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {
    // TODO Auto-generated method stub
    return false;
  }

}
