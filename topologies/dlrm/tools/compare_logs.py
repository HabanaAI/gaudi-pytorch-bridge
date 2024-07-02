import sys
import matplotlib.pyplot as plt
import numpy as np

def crappyhistogram(a, bins=50, width=100):
    h, b = np.histogram(a, bins)

    for i in range (0, bins):
        print('{:12.6f}  | {:{width}s} {}'.format(
            b[i],
            '#'*int(width*h[i]/np.amax(h)),
            h[i],
            width=width))
    print('{:12.6f}  |'.format(b[bins]))

def getValues(log,criterion):

    fetched_values = np.array([])
    found_values = 0
    for line in open(log):
        if line.find("Finished training it") != -1:
            found_values += 1
            words = line.split()
            idx = words.index(criterion)
            fetched_values = np.append(fetched_values,float(words[idx+1].rstrip(",")))


    print('Collected {} values in {}'.format(found_values,log))
    return fetched_values

def compare_logs():
    if (len(sys.argv)!=7):
        print(" USAGE: \n\
                EXAMPLE: python compare_logs.py loss numValuesToCompare atol rtol ref.txt tst_log.txt")
        sys.exit(1)
    else:
        ref_log = sys.argv[5]
        tst_log = sys.argv[6]
        #loss, accuracy
        criterion = sys.argv[1]
        Ylabel = str(criterion)
        Xlabel = 'iterations'
        numVal = sys.argv[2]
        atol = float(sys.argv[3])
        rtol = float(sys.argv[4])
        print('Comparing {} values between {} with {} with atol={} & rtol={} '.format(numVal,ref_log,tst_log,atol,rtol))

    ref = getValues(ref_log,criterion)
    tst = getValues(tst_log,criterion)

    numVal = min(int(numVal),len(tst),len(ref))
    diff = np.abs(ref[0:numVal] - tst[0:numVal])
    max_diff = max(abs(diff))

    numEqual = np.count_nonzero(np.isclose(diff,0*atol,atol=1e-8)==True)
    numDiff1 = np.count_nonzero(np.isclose(diff,1*atol,atol=1e-8)==True)
    numDiff2 = np.count_nonzero(np.isclose(diff,2*atol,atol=1e-8)==True)
    numDiff3 = np.count_nonzero(np.isclose(diff,3*atol,atol=1e-8)==True)
    numDiff4 = np.count_nonzero(np.isclose(diff,4*atol,atol=1e-8)==True)
    numRest = numVal - numDiff1 - numDiff2 - numDiff3 - numDiff4 - numEqual

    indDiff = np.where(np.isclose(diff,0*atol,atol=1e-8)==False)
    indMaxDiff = np.where(np.isclose(abs(diff),max_diff,atol=1e-8)==True)

    print('--------Statistics of differences--------')
    print('Result of allclose(ref,tst,atol,rtol):' , np.allclose(ref[0:numVal] , tst[0:numVal],atol=atol,rtol=rtol))
    print('Equal:',numEqual)
    print('diff=1*atol:',numDiff1)
    print('diff=2*atol:',numDiff2)
    print('diff=3*atol:',numDiff3)
    print('diff=3*atol:',numDiff4)
    print('diff=Larger > 3*atol  :',numRest)


    with np.printoptions(threshold=np.inf):
        print('Iterations of any mismatch :',np.array(indDiff)+1)

    print('Max Abs(Diff) @ iteration=',max_diff,np.array(indMaxDiff)+1)
    print('--------Histogram of differences--------')
    histo,bins = np.histogram(diff,np.arange(0*atol,11*atol,atol))
    print(histo)
    print(bins)

    print('--------Crappy Histogram of differences--------')
    crappyhistogram(diff,bins=len(bins))

    plt.hist(diff)
    plt.title("Histogram of errors")
    plt.ylabel('Frequency')
    plt.xlabel('abs diff(' + criterion + ')  b/w ref and tst')
    plt.savefig("dlrm_error_histo.png")
    plt.close()


    print('--------Plotting the logs-------')

    plt.plot(np.arange(0,numVal,1),ref[0:numVal],label='ref')
    plt.plot(np.arange(0,numVal,1),tst[0:numVal],label='tst')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel(criterion)
    plt.savefig('dlrm_plot.png')
    plt.close()
    print('Plot saved as dlrm_plot.png')

    plt.plot(np.arange(0,numVal,1),diff[0:numVal],label='abs(ref-tst)')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('abs(ref-tst)')
    plt.savefig('dlrm_error_plot.png')
    plt.close()
    print('Plot saved as dlrm_error_plot.png')
    print(f'Command to compare: vimdiff {ref_log} {tst_log}')
if __name__ == "__main__":
    print(sys.argv)
    compare_logs()
