# CS483

Applied Parallel Programming - UIUC Fall 2018

ECE 408, UIUC, Fall 2018

### MXNETproject

Details: https://github.com/illinois-impact/ece408_project

Some rai commands:

```
./rai -p ../ece408_project/ --queue rai_amd64_ece408
./rai -p ../ece408_project/ --queue rai_amd64_ece408 --submit=m1
./rai -p ./ ranking
./rai -p ./ buildtime
./rai -p ./ submitted
./rai -p ./ history
```

To run nvprof:

```
ssh -Y <netid>@linux.ews.illinois.edu
wget <EWS path>
tar –xvf <build.tar.gz>
~/software/cuda-10.0/bin/nvvp &
```

To import the files on GUI NVIDIA profiler:
- File > import > select nvprof > next > single process > next
- timeline data file should be your timeline.nvprof
- event/metrics data file should be your analysis.nvprof.
- finish
