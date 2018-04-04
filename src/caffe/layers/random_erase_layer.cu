#include "caffe/layers/random_erase_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void RandomEraseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
  {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    caffe_copy(count, bottom_data, top_data);
    // if (this->phase_ == TRAIN)
    {
      int batch_size = bottom[0]->shape(0);
      int C = bottom[0]->shape(1);
      int H = bottom[0]->shape(2);
      int W = bottom[0]->shape(3);
      std::cout << bottom[0]->shape(1)
      <<std::endl;
      srand( (unsigned)time( NULL ) );     
      for (int i=0; i < batch_size; i++) {
        // generate W,H
        float r;
        int width = int(rand()%1000/1000*(width_upper_ - width_lower_ + 1)+width_lower_);
        r = rand()%1000/1000.0*(ratio_upper_-ratio_lower_)+ ratio_lower_;
        int height = width * r;
        // generate X,Y
         int Y = rand()%(H - height + 1);
        int X = rand()%(W - width + 1);
        // fill the noise mask
        noise_.Reshape(1,height,width,C);
        int count_noise = noise_.count();
        filler_-> Fill(&noise_);
        Dtype* noise_data = noise_.mutable_gpu_data();
  
        // truncate
        if (truncate_) {
          for (int i = 0; i< count_noise; i++) {
            if (noise_data[i] > trunc_upper_) {
              noise_data[i] = trunc_upper_;
            }
            if (noise_data[i] < trunc_lower_) {
              noise_data[i] = trunc_lower_;
            }
          }
        }
  
        // copy the noise mask to image
        for (int c = 0; c<C; c++) {
          for (int y = Y; y< Y+height; y++) {
            std::cout<<"y: "<<y<<" x: "<<X<<" c: "<<c<< std::endl;
            caffe_copy(width, noise_data + c*height*width + (y-Y)*width,
              top_data + i*C*H*W + c*H*W + y*W + X);
          }
        }
      }
    }
  }

template <typename Dtype>
void RandomEraseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);

}