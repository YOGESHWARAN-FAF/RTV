import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)  # concatenate along channel axis
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, output__channels, kernel_size=5, num_layers=2):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = output__channels

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else output__channels
            self.cells.append(ConvLSTMCell(in_channels, output__channels, kernel_size))

        self.c = None
        self.h = None

    def forward(self, input_seq):
        # input_seq: (batch, seq_len, channels, height, width)
        batch_size, seq_len, _, height, width = input_seq.size()
        if self.c is None:
            self.h, self.c = self.init_hidden(batch_size, height, width)

        outputs = []
        for t in range(seq_len):
            x = input_seq[:, t]
            for i, cell in enumerate(self.cells):
                self.h[i], self.c[i] = cell(x, self.h[i], self.c[i])
                x = self.h[i]
            outputs.append(self.h[-1])

        return torch.stack(outputs, dim=1)

    def inference_forward(self, input_frame):
        # input_seq: (batch, seq_len, channels, height, width)
        batch_size, _, height, width = input_frame.size()
        if self.c is None:
            self.h, self.c = self.init_hidden(batch_size, height, width)


        x = input_frame
        for i, cell in enumerate(self.cells):
            self.h[i], self.c[i] = cell(x, self.h[i], self.c[i])
            x = self.h[i]

        return x[-1].unsqueeze(0)


    def reset(self):
        self.c = None
        self.h = None

    def init_hidden(self, batch_size, height, width):
        p = next(self.parameters())
        h = [torch.zeros(batch_size, self.hidden_channels, height, width, device=p.device, dtype=p.dtype)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_channels, height, width, device=p.device, dtype=p.dtype)
             for _ in range(self.num_layers)]
        return h, c


if __name__ == '__main__':
    # Dummy input: batch of 5 sequences, each with 10 frames of 1-channel 64x64 images
    input_tensor = torch.randn(5, 10, 1, 64, 64)
    input_tensor2 = torch.randn(5, 1, 64, 64)

    model = ConvLSTM(input_channels=1, hidden_channels=16, kernel_size=3, num_layers=1)
    output = model(input_tensor)
    output2 = model.inference_forward(input_tensor2)

    print(output2.shape)  # Expected: (5, 10, 16, 64, 64)
