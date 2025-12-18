#pragma once

#include <iostream>
#include <fstream>
#include <streambuf>

namespace mm_rec {

class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* sb1, std::streambuf* sb2) : sb1_(sb1), sb2_(sb2) {}

protected:
    virtual int overflow(int c) override {
        if (c == EOF) {
            return !EOF;
        } else {
            int const r1 = sb1_->sputc(c);
            int const r2 = sb2_->sputc(c);
            return r1 == EOF || r2 == EOF ? EOF : c;
        }
    }

    virtual int sync() override {
        int const r1 = sb1_->pubsync();
        int const r2 = sb2_->pubsync();
        return r1 == 0 && r2 == 0 ? 0 : -1;
    }

private:
    std::streambuf* sb1_;
    std::streambuf* sb2_;
};

class TeeStream : public std::ostream {
public:
    TeeStream(std::ostream& os1, std::ostream& os2)
        : std::ostream(&tbuf_), tbuf_(os1.rdbuf(), os2.rdbuf()) {}

private:
    TeeBuf tbuf_;
};

} // namespace mm_rec
