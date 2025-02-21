# Differential Gaussian Rasterization used for compression

### Supported features:

- [x] LightGaussian

- [x] Depth rendering (forward), depth supervision (backward)

- [ ] Normal rendering (forward), normal supervision (backward). Ref: [GaussianPro](https://arxiv.org/pdf/2402.14650.pdf)

- [ ] Camera pose (backward)


### How to use

```sh
cp -r diff-gaussian-rasterization Collaborative-NeRF/submodules
cd Collaborative-NeRF
pip install submodules/diff-gaussian-rasterization
```
