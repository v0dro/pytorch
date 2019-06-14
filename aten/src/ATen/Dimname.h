#pragma once
#ifdef NAMEDTENSOR_ENABLED

#include <ATen/core/interned_strings.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace at {

enum class NameType: uint8_t { NORMAL, WILDCARD, TAGGED };

struct CAFFE2_API Dimname {
  static Dimname fromSymbol(Symbol name);
  static Dimname wildcard();

  NameType type() const { return type_; }
  Symbol full_name() const { return full_name_; }
  Symbol untagged_name() const { return untagged_name_; }

 private:
  Dimname(Symbol name)
    : untagged_name_(name), full_name_(name), type_(NameType::NORMAL) {}
  Dimname(NameType type, Symbol full_name, Symbol untagged_name)
    : untagged_name_(untagged_name), full_name_(full_name), type_(type) {}

  // [Dimname Terminology]
  //
  // For "C.in":
  // - "C.in" is the "full name"
  // - "C" is the "untagged name"
  // - "in" is the "tag"
  Symbol untagged_name_;
  Symbol full_name_;
  NameType type_;
  // Will need more fields for other special name types.
};

using DimnameList = c10::ArrayRef<Dimname>;

static Symbol kWildcard = Symbol::dimname("*");
bool CAFFE2_API is_valid_identifier(const std::string& name);

CAFFE2_API c10::optional<Dimname> unify(Dimname dimname, Dimname other);
CAFFE2_API bool match(Dimname dimname, Dimname other);

} // namespace at
#endif
