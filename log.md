- 2020.03.09: Changes dcn to mmdetection version, changed setup.py installation steps, but more doesn't converge that is a big problem.

- dcn head and deconv head not same:

  dcn:

  ```
  INFO 03.08 20:20:59 centernet_deconv_dc.py:48: in x shpe: torch.Size([4, 2048, 16, 16])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:50: dcn out shpe: torch.Size([4, 256, 16, 16])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:52: dcn_bn out shpe: torch.Size([4, 256, 16, 16])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:48: in x shpe: torch.Size([4, 256, 32, 32])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:50: dcn out shpe: torch.Size([4, 128, 32, 32])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:52: dcn_bn out shpe: torch.Size([4, 128, 32, 32])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:48: in x shpe: torch.Size([4, 128, 64, 64])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:50: dcn out shpe: torch.Size([4, 64, 64, 64])
  INFO 03.08 20:21:01 centernet_deconv_dc.py:52: dcn_bn out shpe: torch.Size([4, 64, 64, 64])

  ```

  

  deconv:

  ```
  INFO 03.08 20:14:37 centernet_deconv.py:50: in x shpe: torch.Size([4, 2048, 16, 16])
  INFO 03.08 20:14:37 centernet_deconv.py:52: dcn out shpe: torch.Size([4, 256, 14, 14])
  INFO 03.08 20:14:37 centernet_deconv.py:54: dcn_bn out shpe: torch.Size([4, 256, 14, 14])
  INFO 03.08 20:14:37 centernet_deconv.py:50: in x shpe: torch.Size([4, 256, 28, 28])
  INFO 03.08 20:14:37 centernet_deconv.py:52: dcn out shpe: torch.Size([4, 128, 26, 26])
  INFO 03.08 20:14:37 centernet_deconv.py:54: dcn_bn out shpe: torch.Size([4, 128, 26, 26])
  INFO 03.08 20:14:37 centernet_deconv.py:50: in x shpe: torch.Size([4, 128, 52, 52])
  INFO 03.08 20:14:37 centernet_deconv.py:52: dcn out shpe: torch.Size([4, 64, 50, 50])
  INFO 03.08 20:14:37 centernet_deconv.py:54: dcn_bn out shpe: torch.Size([4, 64, 50, 50])
  ```

  