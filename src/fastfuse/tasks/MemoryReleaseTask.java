package fastfuse.tasks;

import java.util.Arrays;
import java.util.List;

import clearcl.ClearCLImage;
import fastfuse.FastFusionEngineInterface;
import fastfuse.FastFusionMemoryPool;

public class MemoryReleaseTask extends TaskBase
                               implements TaskInterface
{

  private final String[] mImageKeysToRelease;

  public MemoryReleaseTask(String pImageKeyRequired,
                           String... pImageKeysToRelease)
  {
    this(Arrays.asList(pImageKeyRequired), pImageKeysToRelease);
  }

  public MemoryReleaseTask(List<String> pImageKeysRequired,
                           String... pImageKeysToRelease)
  {
    super((String[]) pImageKeysRequired.toArray());
    mImageKeysToRelease = pImageKeysToRelease;
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {
    for (String lImageKey : mImageKeysToRelease)
    {
      ClearCLImage lImage = pFastFusionEngine.getImage(lImageKey);
      FastFusionMemoryPool.get().releaseImage(lImageKey, lImage);
    }
    return true;
  }

}
