import matplotlib.pyplot as plt

def Curve(epoch,prediction,y_test):


    assert prediction.shape == y_test.shape

    x = [i for i in range(y_test.shape[1])]
    for c in range(int(prediction.shape[2])):
        plt.rcParams['figure.figsize'] = (10, 10)
        outputs = prediction [0, :, c].cpu().detach().numpy()
        batch_y = y_test [0, :, c].cpu().detach().numpy()

        plt.plot(x, batch_y ,c = 'b')
        plt.plot(x, outputs ,c = 'r')


        plt.savefig('Figs/epoch{}_result_{}.jpg'.format(epoch+1, c))
        plt.clf()

