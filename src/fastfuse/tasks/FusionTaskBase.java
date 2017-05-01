package fastfuse.tasks;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import clearcl.ClearCLContext;
import clearcl.ClearCLKernel;
import clearcl.ClearCLProgram;
import fastfuse.FastFusionEngineInterface;

/**
 * Base class providing common fields and methods for all task implementations
 *
 * @author royer
 */
public abstract class FusionTaskBase implements FusionTaskInterface
{

  private final HashSet<String> mRequiredImagesSlotKeysSet =
                                                           new HashSet<>();

  private Class<AverageTask> mClass;
  private String mSourceFile;
  private ClearCLProgram mProgram;
  private HashMap<String, ClearCLKernel> mKernelMap =
                                                    new HashMap<String, ClearCLKernel>();

  /**
   * Instantiates a fusion task given the keys of required images
   * 
   * @param pSlotKeys
   *          list of slot keys
   */
  public FusionTaskBase(String... pSlotKeys)
  {
    super();
    for (String lSlotKey : pSlotKeys)
      mRequiredImagesSlotKeysSet.add(lSlotKey);
  }

  protected void setupProgram(Class<AverageTask> pClass,
                              String pSourceFile)
  {
    mClass = pClass;
    mSourceFile = pSourceFile;
  }

  protected ClearCLKernel getKernel(ClearCLContext pContext,
                                    String pKernelName) throws IOException
  {
    if (mKernelMap.get(pKernelName) != null)
      return mKernelMap.get(pKernelName);
    mProgram = pContext.createProgram(mClass, mSourceFile);
    mProgram.addBuildOptionAllMathOpt();
    mProgram.buildAndLog();
    ClearCLKernel lKernel = mProgram.createKernel(pKernelName);
    mKernelMap.put(pKernelName, lKernel);
    return lKernel;
  }

  @Override
  public boolean checkIfRequiredImagesAvailable(Set<String> pAvailableImagesSlotKeys)
  {
    boolean lAllRequiredImagesAvailable =
                                        pAvailableImagesSlotKeys.containsAll(mRequiredImagesSlotKeysSet);

    return lAllRequiredImagesAvailable;
  }

  @Override
  public abstract boolean enqueue(FastFusionEngineInterface pStackFuser,
                                  boolean pWaitToFinish);

  @Override
  public String toString()
  {
    return String.format("FusionTaskBase [mRequiredImagesSlotKeysSet=%s, mKernels=%s]",
                         mRequiredImagesSlotKeysSet,
                         mKernelMap.toString());
  }

}
