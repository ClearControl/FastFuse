package fastfuse.tasks;

import java.util.HashMap;
import java.util.Map;

import clearcl.ClearCLImage;
import clearcl.enums.ImageChannelDataType;

public class TaskHelper
{
  private static boolean allowedDataType(ImageChannelDataType pDataType)
  {
    return pDataType == ImageChannelDataType.Float
           || pDataType == ImageChannelDataType.UnsignedInt16;
  }

  public static boolean allowedDataType(ImageChannelDataType... pDataTypes)
  {
    if (pDataTypes == null)
      return true;
    for (ImageChannelDataType lDataType : pDataTypes)
      if (!allowedDataType(lDataType))
        return false;
    return true;
  }

  public static boolean allowedDataType(ClearCLImage... pImages)
  {
    if (pImages == null)
      return true;
    ImageChannelDataType[] dataTypes =
                                     new ImageChannelDataType[pImages.length];
    for (int i = 0; i < pImages.length; i++)
      dataTypes[i] = pImages[i].getChannelDataType();
    return allowedDataType(dataTypes);
  }

  public static Map<String, Object> getOpenCLDefines(ImageChannelDataType pDTypeIn,
                                                     ImageChannelDataType pDTypeOut)
  {
    assert allowedDataType(pDTypeIn, pDTypeOut);
    Map<String, Object> lDefines = new HashMap<>();
    lDefines.put("DTYPE_IN",
                 pDTypeIn.isInteger() ? "ushort" : "float");
    lDefines.put("DTYPE_OUT",
                 pDTypeOut.isInteger() ? "ushort" : "float");
    lDefines.put("READ_IMAGE",
                 pDTypeIn.isInteger() ? "read_imageui"
                                      : "read_imagef");
    lDefines.put("WRITE_IMAGE",
                 pDTypeOut.isInteger() ? "write_imageui"
                                       : "write_imagef");
    return lDefines;
  }

  public static Map<String, Object> getOpenCLDefines(ClearCLImage pImageIn,
                                                     ClearCLImage pImageOut)
  {
    return getOpenCLDefines(pImageIn.getChannelDataType(),
                            pImageOut.getChannelDataType());
  }

}
