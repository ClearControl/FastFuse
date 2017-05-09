__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float hx[] = {-1,-2,-1,-2,-4,-2,-1,-2,-1,0,0,0,0,0,0,0,0,0,1,2,1,2,4,2,1,2,1};
__constant float hy[] = {-1,-2,-1,0,0,0,1,2,1,-2,-4,-2,0,0,0,2,4,2,-1,-2,-1,0,0,0,1,2,1};
__constant float hz[] = {-1,0,1,-2,0,2,-1,0,1,-2,0,2,-4,0,4,-2,0,2,-1,0,1,-2,0,2,-1,0,1};


inline float sobel_magnitude_squared_ui(read_only image3d_t src, const int i0, const int j0, const int k0) {
  float Gx = 0.0f, Gy = 0.0f, Gz = 0.0f;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 3; ++k) {
    const int dx = i-1, dy = j-1, dz = k-1;
    const int ind = i + 3*j + 3*3*k;
    const ushort src_val = read_imageui(src,sampler,(int4)(i0+dx,j0+dy,k0+dz,0)).x;
    Gx += hx[ind]*src_val;
    Gy += hy[ind]*src_val;
    Gz += hz[ind]*src_val;
  }
  return Gx*Gx + Gy*Gy + Gz*Gz;
}


inline float sobel_magnitude_squared_f(read_only image3d_t src, const int i0, const int j0, const int k0) {
  float Gx = 0.0f, Gy = 0.0f, Gz = 0.0f;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 3; ++k) {
    const int dx = i-1, dy = j-1, dz = k-1;
    const int ind = i + 3*j + 3*3*k;
    const float src_val = read_imagef(src,sampler,(int4)(i0+dx,j0+dy,k0+dz,0)).x;
    Gx += hx[ind]*src_val;
    Gy += hy[ind]*src_val;
    Gz += hz[ind]*src_val;
  }
  return Gx*Gx + Gy*Gy + Gz*Gz;
}


__kernel void fuse_4_imageui_to_imagef(write_only image3d_t dst, read_only image3d_t src1, read_only image3d_t src2, read_only image3d_t src3, read_only image3d_t src4) {

  const int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  const int4 coord = (int4)(i,j,k,0);

  float w1 = sobel_magnitude_squared_ui(src1,i,j,k);
  float w2 = sobel_magnitude_squared_ui(src2,i,j,k);
  float w3 = sobel_magnitude_squared_ui(src3,i,j,k);
  float w4 = sobel_magnitude_squared_ui(src4,i,j,k);

  const float wsum = w1 + w2 + w3 + w4 + 1e-30; // add small epsilon to avoid wsum = 0
  w1 /= wsum;  w2 /= wsum;  w3 /= wsum;  w4 /= wsum;

  const ushort v1 = read_imageui(src1,sampler,coord).x;
  const ushort v2 = read_imageui(src2,sampler,coord).x;
  const ushort v3 = read_imageui(src3,sampler,coord).x;
  const ushort v4 = read_imageui(src4,sampler,coord).x;
  const float res = w1*v1 + w2*v2 + w3*v3 + w4*v4;

  write_imagef(dst,coord,res);
}


__kernel void fuse_2_imagef_to_imageui(write_only image3d_t dst, read_only image3d_t src1, read_only image3d_t src2) {

  const int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  const int4 coord = (int4)(i,j,k,0);

  float w1 = sobel_magnitude_squared_f(src1,i,j,k);
  float w2 = sobel_magnitude_squared_f(src2,i,j,k);

  const float wsum = w1 + w2 + 1e-30; // add small epsilon to avoid wsum = 0
  w1 /= wsum;  w2 /= wsum;

  const float v1  = read_imagef(src1,sampler,coord).x;
  const float v2  = read_imagef(src2,sampler,coord).x;
  const float res = w1*v1 + w2*v2;

  write_imageui(dst,coord,(ushort)res);
}
